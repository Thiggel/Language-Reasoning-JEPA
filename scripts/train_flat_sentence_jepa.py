"""Train the single-level token-action causal sentence-state JEPA."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    from train_sentence_hierarchy import (
        _pairwise_advantage_loss, dense_loss, make_dataset, normalized_mse,
        vicreg,
    )
except ModuleNotFoundError:
    from scripts.train_sentence_hierarchy import (
        _pairwise_advantage_loss, dense_loss, make_dataset, normalized_mse,
        vicreg,
    )
from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.sampling import FreshEpochSampler
from textjepa.data.semantic_lm import collate_semantic_lm
from textjepa.models.flat_sentence_jepa import FlatSentenceJEPA
from textjepa.training.loggers import MetricLogger
from textjepa.training.optim import build_optimizer, cosine_warmup, ema_momentum
from textjepa.utils import seed_everything
from textjepa.utils.metrics import effective_rank, feature_std


def forward(model, batch, device):
    return model(batch["tokens"].to(device), batch["prompt_len"].to(device))


def compute_losses(out, cfg, model, batch):
    obj = cfg.objective
    prediction = normalized_mse(out["pred"], out["target"], out["valid"])
    dense = dense_loss(
        out["dense_predictions"], out["dense_targets"],
        out["dense_masks"], obj.dense_discount,
    )
    goal = F.smooth_l1_loss(
        F.layer_norm(out["goal_pred"], out["goal_pred"].shape[-1:]),
        F.layer_norm(out["final_target"].detach(), out["final_target"].shape[-1:]),
    )
    prior = prediction.sum() * 0.0
    prior_accuracy = prior.detach()
    if out["token_prior_logits"] is not None:
        logits = out["token_prior_logits"][out["valid"]]
        labels = out["action_ids"][out["valid"]]
        prior = F.cross_entropy(logits, labels)
        prior_accuracy = logits.argmax(-1).eq(labels).float().mean()
    regularizer = vicreg(out["prev"][out["valid"]], obj.covariance)
    cf = model.token_counterfactuals(
        out, batch["tokens"].to(out["states"].device),
        batch["prompt_len"].to(out["states"].device), k=int(obj.gar_k),
        max_anchors=int(obj.gar_max_anchors),
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
    total = (
        obj.prediction * prediction + obj.dense * dense
        + obj.token_prior * prior + obj.goal * goal
        + obj.vicreg * regularizer + obj.gar * gar_total
    )
    items = {
        "prediction": prediction, "dense": dense, "token_prior": prior,
        "token_prior_accuracy": prior_accuracy, "goal": goal,
        "vicreg": regularizer, "gar_regression": gar_regression,
        "gar_ranking": gar_ranking, "gar_counterfactual_mse": gar_dynamics,
        "gar_advantage_std": cf["advantage_target"].std(),
        "selection": prediction.detach() + dense.detach() + prior.detach(),
    }
    return total, items


@hydra.main(config_path="../configs", config_name="flat_sentence_jepa", version_base="1.3")
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
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers, collate_fn=collate,
    )
    model = FlatSentenceJEPA(len(vocab), vocab.pad_id, **cfg.model).to(cfg.device)
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
            loss, items = compute_losses(out, cfg, model, batch)
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
        sums, count, features = {}, 0, []
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if index >= cfg.train.eval_batches:
                    break
                out = forward(model, batch, cfg.device)
                _, items = compute_losses(out, cfg, model, batch)
                for key, value in items.items():
                    sums[key] = sums.get(key, 0.0) + float(value)
                features.append(out["target"][out["valid"]])
                count += 1
        metrics = {key: value / count for key, value in sums.items()}
        feature = torch.cat(features)
        metrics.update(
            state_std=feature_std(feature),
            state_rank=effective_rank(feature[:4096]),
        )
        logger.log(step, metrics, prefix="val/")
        payload = {
            "model": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
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
