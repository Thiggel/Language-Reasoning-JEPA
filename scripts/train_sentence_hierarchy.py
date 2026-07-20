"""Train the two-level, distinct-space token/sentence JEPA."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    from train_token_hierarchy_v2 import dense_loss, masked_mean, normalized_mse, vicreg
except ModuleNotFoundError:  # imported as ``scripts.*`` by tests
    from scripts.train_token_hierarchy_v2 import dense_loss, masked_mean, normalized_mse, vicreg
from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.sampling import FreshEpochSampler
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset, collate_semantic_lm
from textjepa.models.sentence_hierarchy import SentenceHierarchyJEPA
from textjepa.training.loggers import MetricLogger
from textjepa.training.optim import build_optimizer, cosine_warmup, ema_momentum
from textjepa.utils import seed_everything
from textjepa.utils.metrics import effective_rank, feature_std


def make_dataset(cfg, vocab, size, seed):
    return SemanticBoundaryLMDataset(
        vocab, size=size, seed=seed, boundary_mode=cfg.boundary_mode,
        modulus=cfg.data.modulus, n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob, steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )


def forward(model, batch, device):
    return model(
        batch["tokens"].to(device), batch["prompt_len"].to(device),
        batch["sentence_ends"].to(device),
    )


def _pairwise_advantage_loss(value, target, margin):
    better = target.unsqueeze(2) > target.unsqueeze(1)
    loss = F.relu(margin - value.unsqueeze(2) + value.unsqueeze(1))
    return loss[better].mean() if better.any() else loss.sum() * 0.0


def _temporal_straightening(model, level):
    projected = model.planning_projection(level["target"])
    valid = level["valid"][:, 2:]
    if not valid.any():
        return projected.sum() * 0.0
    second = projected[:, 2:] - 2 * projected[:, 1:-1] + projected[:, :-2]
    return masked_mean(second.square().mean(-1), valid.float())


def _value_monotonicity(level, margin):
    value = level["value"]
    valid = level["valid"][:, 1:] & level["valid"][:, :-1]
    if not valid.any():
        return value.sum() * 0.0
    # Value is remaining cost and should decrease at a completed sentence.
    violation = F.relu(margin + value[:, 1:] - value[:, :-1])
    return masked_mean(violation, valid.float())


def compute_sentence_losses(out, cfg, model, batch):
    obj = cfg.objective
    level = out["sentence_level"]
    low = normalized_mse(out["low_pred"], out["target"], out["valid"])
    low_dense = dense_loss(
        out["low_dense_predictions"], out["low_dense_targets"],
        out["low_dense_masks"], obj.dense_discount,
    )
    low_value = masked_mean(F.smooth_l1_loss(
        out["low_value"], out["low_remaining_target"], reduction="none"
    ), out["valid"].float())
    low_goal = F.smooth_l1_loss(
        F.layer_norm(out["goal_pred"], out["goal_pred"].shape[-1:]),
        F.layer_norm(out["final_target"].detach(), out["final_target"].shape[-1:]),
    )
    token_prior = out["low_pred"].sum() * 0.0
    token_prior_accuracy = token_prior.detach()
    if out["token_prior_logits"] is not None:
        logits = out["token_prior_logits"][out["valid"]]
        labels = out["action_ids"][out["valid"]]
        token_prior = F.cross_entropy(logits, labels)
        token_prior_accuracy = logits.argmax(-1).eq(labels).float().mean()
    low_states = out["prev"][out["valid"]]
    low_regularizer = vicreg(low_states, obj.covariance)

    high = normalized_mse(level["pred"], level["target"], level["valid"])
    high_dense = dense_loss(
        level["dense_predictions"], level["dense_targets"],
        level["dense_masks"], obj.dense_discount,
    )
    high_goal = F.smooth_l1_loss(
        F.layer_norm(out["high_goal_pred"], out["high_goal_pred"].shape[-1:]),
        F.layer_norm(out["high_final_target"].detach(), out["high_final_target"].shape[-1:]),
    )
    high_value = masked_mean(F.smooth_l1_loss(
        level["value"], level["remaining_target"], reduction="none"
    ), level["valid"].float())
    high_states = level["prev"][level["valid"]]
    high_regularizer = vicreg(high_states, obj.covariance)
    # Full Gaussian negative log likelihood is bounded and calibrated. The
    # legacy deterministic surrogate omits its constant and may become
    # negative, which makes it unsafe as an optimization objective.
    macro_prior = masked_mean(level["prior_nll"], level["valid"].float())
    support = masked_mean(
        F.softplus(-level["support_pos"]) + F.softplus(level["support_neg"]),
        level["valid"].float(),
    )
    bridge = normalized_mse(
        level["low_endpoint_high"], level["target"], level["valid"]
    ) + normalized_mse(
        level["high_target_low"], level["low_target"], level["valid"]
    )
    transition_reachability = normalized_mse(
        level["pred"], level["low_endpoint_high"].detach(), level["valid"]
    )
    positive = model.reachability(level["low_start"], level["target"].detach())
    negative = model.reachability(
        level["low_start"], level["target"].detach().roll(1, 0)
    )
    reachability_classifier = masked_mean(
        F.softplus(-positive) + F.softplus(negative), level["valid"].float()
    )
    straightening = _temporal_straightening(model, level)
    monotonicity = _value_monotonicity(level, float(obj.monotonicity_margin))

    low_total = (
        obj.low_prediction * low + obj.low_dense * low_dense
        + obj.low_value * low_value + obj.low_goal * low_goal
        + obj.token_prior * token_prior + obj.vicreg_low * low_regularizer
    )
    high_total = (
        obj.high_prediction * high + obj.high_dense * high_dense
        + obj.high_goal * high_goal + obj.high_value * high_value
        + obj.macro_prior * macro_prior + obj.support * support
        + obj.bridge * bridge
        + obj.transition_reachability * transition_reachability
        + obj.reachability_classifier * reachability_classifier
        + obj.vicreg_high * high_regularizer
        + obj.temporal_straightening * straightening
        + obj.value_monotonicity * monotonicity
    )
    items = {
        "low_prediction": low, "low_dense": low_dense,
        "low_value": low_value, "low_goal": low_goal,
        "token_prior": token_prior,
        "token_prior_accuracy": token_prior_accuracy,
        "low_vicreg": low_regularizer, "high_prediction": high,
        "high_dense": high_dense, "high_goal": high_goal,
        "high_value": high_value, "high_vicreg": high_regularizer,
        "macro_prior": macro_prior, "support": support, "bridge": bridge,
        "transition_reachability": transition_reachability,
        "reachability_classifier": reachability_classifier,
        "temporal_straightening": straightening,
        "value_monotonicity": monotonicity,
    }
    gar_total = low_total.sum() * 0.0
    if float(obj.gar_weight) > 0:
        cf = model.sentence_counterfactuals(
            out, batch["tokens"].to(out["states"].device),
            batch["prompt_len"].to(out["states"].device),
            k=int(obj.gar_k), source=str(obj.gar_source),
        )
        gar_regression = F.smooth_l1_loss(cf["value"], cf["advantage_target"])
        gar_ranking = _pairwise_advantage_loss(
            cf["value"], cf["advantage_target"], float(obj.gar_margin)
        )
        gar_dynamics = normalized_mse(
            cf["predicted_outcome"], cf["exact_outcome"].detach(),
            cf["candidate_valid"],
        )
        gar_total = (
            obj.gar_regression * gar_regression
            + obj.gar_ranking * gar_ranking
            + obj.gar_counterfactual_mse * gar_dynamics
        )
        items.update(
            gar_regression=gar_regression, gar_ranking=gar_ranking,
            gar_counterfactual_mse=gar_dynamics,
            gar_advantage_std=cf["advantage_target"].std(),
        )
    total = low_total + obj.high_level_weight * high_total + obj.gar_weight * gar_total
    selection = (
        low.detach() + low_dense.detach() + token_prior.detach()
        + obj.high_level_weight * (high.detach() + high_dense.detach())
    )
    items.update(loss=total, selection=selection)
    return total, items


@hydra.main(config_path="../configs", config_name="sentence_hierarchy", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))
    vocab = build_vocab(cfg.data.modulus)
    train_ds = make_dataset(cfg, vocab, cfg.data.train_size, cfg.data.train_seed)
    val_ds = make_dataset(cfg, vocab, cfg.data.val_size, cfg.data.val_seed)
    sampler = FreshEpochSampler(train_ds, seed=cfg.seed)
    collate = partial(collate_semantic_lm, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
        num_workers=cfg.train.num_workers, collate_fn=collate, drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers, collate_fn=collate,
    )
    model = SentenceHierarchyJEPA(
        len(vocab), vocab.pad_id, **cfg.model
    ).to(cfg.device)
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    total_steps = cfg.train.epochs * len(train_loader)
    logger = MetricLogger(out_dir)
    step, best = 0, float("inf")
    for epoch in range(cfg.train.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            for group in optimizer.param_groups:
                group["lr"] = cfg.train.lr * cosine_warmup(
                    step, total_steps, cfg.train.warmup_steps
                )
            out = forward(model, batch, cfg.device)
            loss, items = compute_sentence_losses(out, cfg, model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            model.update_teacher(ema_momentum(
                step, total_steps, cfg.train.ema_start, cfg.train.ema_end
            ))
            if step % cfg.train.log_every == 0:
                logger.log(step, {
                    key: float(value.detach()) for key, value in items.items()
                }, prefix="train/")
            step += 1
        model.eval()
        sums, count, low_features, high_features, action_features = {}, 0, [], [], []
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if index >= cfg.train.eval_batches:
                    break
                out = forward(model, batch, cfg.device)
                _, items = compute_sentence_losses(out, cfg, model, batch)
                for key, value in items.items():
                    sums[key] = sums.get(key, 0.0) + float(value)
                low_features.append(out["prev"][out["valid"]])
                level = out["sentence_level"]
                high_features.append(level["target"][level["valid"]])
                action_features.append(level["codes"][level["valid"]])
                count += 1
        metrics = {key: value / count for key, value in sums.items()}
        low_cat, high_cat, action_cat = map(torch.cat, (
            low_features, high_features, action_features
        ))
        metrics.update(
            low_state_std=feature_std(low_cat),
            low_state_rank=effective_rank(low_cat[:4096]),
            high_state_std=feature_std(high_cat),
            high_state_rank=effective_rank(high_cat[:4096]),
            macro_action_std=feature_std(action_cat),
            macro_action_rank=effective_rank(action_cat[:4096]),
        )
        logger.log(step, metrics, prefix="val/")
        payload = {
            "model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch, "metrics": metrics,
        }
        torch.save(payload, out_dir / "last.pt")
        if metrics["selection"] < best:
            best = metrics["selection"]
            torch.save(payload, out_dir / "best.pt")
        print(f"[epoch {epoch}] " + "  ".join(
            f"{key}={value:.4f}" for key, value in sorted(metrics.items())
        ), flush=True)
    logger.close()


if __name__ == "__main__":
    main()
