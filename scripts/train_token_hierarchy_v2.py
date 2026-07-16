"""Train the multilevel causal token-to-span JEPA."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.data.sampling import FreshEpochSampler
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA
from textjepa.training.loggers import MetricLogger
from textjepa.training.optim import build_optimizer, cosine_warmup, ema_momentum
from textjepa.utils import seed_everything
from textjepa.utils.metrics import effective_rank, feature_std


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def normalized_mse(pred, target, mask):
    pred = F.layer_norm(pred, pred.shape[-1:])
    target = F.layer_norm(target, target.shape[-1:])
    return masked_mean((pred - target).square().mean(-1), mask.float())


def dense_loss(predictions, targets, masks, discount):
    losses, weights = [], []
    for horizon, (prediction, target, mask) in enumerate(
        zip(predictions, targets, masks)
    ):
        weight = discount ** horizon
        losses.append(weight * normalized_mse(prediction, target, mask))
        weights.append(weight)
    return torch.stack(losses).sum() / sum(weights)


def vicreg(states, covariance):
    centered = states - states.mean(0)
    std = torch.sqrt(centered.var(0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    cov = centered.T @ centered / max(states.shape[0] - 1, 1)
    cov = cov - torch.diag_embed(torch.diagonal(cov))
    return variance + covariance * cov.square().sum() / states.shape[-1]


def compute_losses(out, cfg):
    obj = cfg.objective
    low = normalized_mse(out["low_pred"], out["target"], out["valid"])
    low_dense = dense_loss(
        out["low_dense_predictions"], out["low_dense_targets"],
        out["low_dense_masks"], obj.dense_discount,
    )
    low_value = masked_mean(
        F.smooth_l1_loss(
            out["low_value"], out["low_remaining_target"], reduction="none"
        ), out["valid"].float(),
    )
    goal = F.smooth_l1_loss(
        F.layer_norm(out["goal_pred"], out["goal_pred"].shape[-1:]),
        F.layer_norm(out["final_target"].detach(), out["final_target"].shape[-1:]),
    )
    state_flat = out["target"].reshape(-1, out["target"].shape[-1])[
        out["valid"].reshape(-1)
    ]
    regularizer = vicreg(state_flat, obj.covariance)
    total = (
        obj.low_prediction * low
        + obj.low_dense * low_dense
        + obj.low_value * low_value
        + obj.goal_prediction * goal
        + obj.vicreg * regularizer
    )
    items = {
        "low_prediction": low,
        "low_dense": low_dense,
        "low_value": low_value,
        "goal_prediction": goal,
        "vicreg": regularizer,
    }
    selection = (
        obj.low_prediction * low.detach()
        + obj.low_dense * low_dense.detach()
        + obj.goal_prediction * goal.detach()
    )
    if out["token_prior_logits"] is not None:
        logits = out["token_prior_logits"][out["valid"]]
        labels = out["action_ids"][out["valid"]]
        token_prior = F.cross_entropy(
            logits,
            labels,
            label_smoothing=float(obj.token_prior_label_smoothing),
        )
        token_prior_accuracy = logits.argmax(-1).eq(labels).float().mean()
        token_prior_entropy = torch.distributions.Categorical(
            logits=logits
        ).entropy().mean()
        total = total + obj.token_prior * token_prior
        selection = selection + obj.token_prior * token_prior.detach()
        items.update({
            "token_prior": token_prior,
            "token_prior_accuracy": token_prior_accuracy,
            "token_prior_entropy": token_prior_entropy,
        })
        rollout_losses = []
        rollout_weights = []
        for horizon, rollout_logits in enumerate(
            out["token_prior_rollout_logits"], 1
        ):
            mask = out["valid"][:, horizon:]
            labels = out["action_ids"][:, horizon:]
            if not mask.any():
                continue
            loss_at_horizon = F.cross_entropy(
                rollout_logits[mask], labels[mask],
                label_smoothing=float(obj.token_prior_label_smoothing),
            )
            weight = float(obj.token_prior_rollout_discount) ** (horizon - 1)
            rollout_losses.append(weight * loss_at_horizon)
            rollout_weights.append(weight)
            items[f"token_prior_rollout_h{horizon}"] = loss_at_horizon
        if rollout_losses:
            token_prior_rollout = torch.stack(rollout_losses).sum() / sum(
                rollout_weights
            )
            total = total + obj.token_prior_rollout * token_prior_rollout
            selection = (
                selection
                + obj.token_prior_rollout * token_prior_rollout.detach()
            )
            items["token_prior_rollout"] = token_prior_rollout
    for level in out["levels"]:
        mask = level["valid"]
        high = normalized_mse(level["pred"], level["target"], mask)
        high_dense = dense_loss(
            level["dense_predictions"], level["dense_targets"],
            level["dense_masks"], obj.dense_discount,
        )
        reachability = normalized_mse(
            level["pred"], level["recursive_low_endpoint"].detach(), mask
        )
        value = masked_mean(F.smooth_l1_loss(
            level["value"], level["remaining_target"], reduction="none"
        ), mask.float())
        prior = masked_mean(level["prior_nll"], mask.float())
        support = masked_mean(
            F.softplus(-level["support_pos"]) + F.softplus(level["support_neg"]),
            mask.float(),
        )
        total = total + (
            obj.high_prediction * high
            + obj.high_dense * high_dense
            + obj.reachability * reachability
            + obj.high_value * value
            + obj.macro_prior * prior
            + obj.support * support
        )
        prefix = f"level{level['index'] + 1}"
        items.update({
            f"{prefix}_prediction": high,
            f"{prefix}_dense": high_dense,
            f"{prefix}_reachability": reachability,
            f"{prefix}_value": value,
            f"{prefix}_prior": prior,
            f"{prefix}_support": support,
        })
        selection = selection + (
            obj.high_prediction * high.detach()
            + obj.high_dense * high_dense.detach()
            + obj.reachability * reachability.detach()
        )
    items["loss"] = total
    items["selection"] = selection
    return total, items


def make_dataset(cfg, vocab, size, seed):
    return LMDataset(
        vocab, size=size, seed=seed, modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )


@hydra.main(config_path="../configs", config_name="token_hierarchy_v2", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))
    vocab = build_vocab(cfg.data.modulus)
    train_ds = make_dataset(cfg, vocab, cfg.data.train_size, cfg.data.train_seed)
    val_ds = make_dataset(cfg, vocab, cfg.data.val_size, cfg.data.val_seed)
    sampler = FreshEpochSampler(train_ds, seed=cfg.seed)
    collate = partial(collate_lm, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
        num_workers=cfg.train.num_workers, collate_fn=collate, drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        collate_fn=collate,
    )
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
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
            out = model(batch["tokens"].to(cfg.device), batch["prompt_len"].to(cfg.device))
            loss, items = compute_losses(out, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            model.update_teacher(ema_momentum(
                step, total_steps, cfg.train.ema_start, cfg.train.ema_end
            ))
            if step % cfg.train.log_every == 0:
                logger.log(step, {k: float(v.detach()) for k, v in items.items()}, prefix="train/")
            step += 1
        model.eval()
        sums, count, state_features, level_features = {}, 0, [], None
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if index >= cfg.train.eval_batches:
                    break
                out = model(batch["tokens"].to(cfg.device), batch["prompt_len"].to(cfg.device))
                _, items = compute_losses(out, cfg)
                for name, value in items.items():
                    sums[name] = sums.get(name, 0.0) + float(value)
                state_features.append(out["target"][out["valid"]])
                if level_features is None:
                    level_features = [[] for _ in out["levels"]]
                for i, level in enumerate(out["levels"]):
                    level_features[i].append(level["codes"][level["valid"]])
                count += 1
        metrics = {name: value / count for name, value in sums.items()}
        states = torch.cat(state_features)
        metrics.update(state_std=feature_std(states), state_rank=effective_rank(states[:4096]))
        for i, features in enumerate(level_features or []):
            codes = torch.cat(features)
            metrics[f"level{i + 1}_action_std"] = feature_std(codes)
            metrics[f"level{i + 1}_action_rank"] = effective_rank(codes[:4096])
        logger.log(step, metrics, prefix="val/")
        payload = {
            "model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch, "metrics": metrics,
        }
        torch.save(payload, out_dir / "last.pt")
        if metrics["selection"] < best:
            best = metrics["selection"]
            torch.save(payload, out_dir / "best.pt")
        print(f"[epoch {epoch}] " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())), flush=True)
    logger.close()


if __name__ == "__main__":
    main()
