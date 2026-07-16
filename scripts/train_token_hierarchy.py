"""Train the fixed-span multiscale token JEPA."""

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
from textjepa.models.token_hierarchy import TokenHierarchyJEPA
from textjepa.training.loggers import MetricLogger
from textjepa.training.optim import (
    build_optimizer,
    cosine_warmup,
    ema_momentum,
)
from textjepa.utils import seed_everything
from textjepa.utils.metrics import effective_rank, feature_std


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1)


def normalized_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    pred = F.layer_norm(pred, pred.shape[-1:])
    target = F.layer_norm(target, target.shape[-1:])
    return masked_mean((pred - target).square().mean(-1), mask.float())


def vicreg(states: torch.Tensor, weight: float) -> torch.Tensor:
    if states.shape[0] < 2:
        return states.sum() * 0.0
    centered = states - states.mean(0)
    std = torch.sqrt(centered.var(0, unbiased=False) + 1e-4)
    std_loss = F.relu(1.0 - std).mean()
    cov = centered.T @ centered / max(states.shape[0] - 1, 1)
    cov = cov - torch.diag_embed(torch.diagonal(cov))
    return std_loss + weight * cov.square().sum() / states.shape[-1]


def losses(out: dict, cfg: DictConfig) -> tuple[torch.Tensor, dict[str, float]]:
    obj = cfg.objective
    low = normalized_mse(out["low_pred"], out["target"], out["valid"])
    high = normalized_mse(
        out["high_pred"], out["high_target"], out["high_valid"]
    )
    prior = masked_mean(
        out["macro_prior_loss"], out["high_valid"].float()
    )
    high_value = masked_mean(
        F.smooth_l1_loss(
            out["high_value"],
            out["high_value_target"].detach() * obj.value_scale,
            reduction="none",
        ),
        out["high_valid"].float(),
    )
    state_flat = out["target"].reshape(-1, out["target"].shape[-1])[
        out["valid"].reshape(-1)
    ]
    reg = vicreg(state_flat, obj.covariance)
    total = (
        obj.low_prediction * low
        + obj.high_prediction * high
        + obj.macro_prior * prior
        + obj.high_value * high_value
        + obj.vicreg * reg
    )
    return total, {
        "loss": float(total.detach()),
        "low_prediction": float(low.detach()),
        "high_prediction": float(high.detach()),
        "macro_prior": float(prior.detach()),
        "high_value": float(high_value.detach()),
        "vicreg": float(reg.detach()),
    }


@hydra.main(
    config_path="../configs", config_name="token_hierarchy", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))
    vocab = build_vocab(cfg.data.modulus)

    def make(size: int, seed: int):
        return LMDataset(
            vocab,
            size=size,
            seed=seed,
            modulus=cfg.data.modulus,
            n_vars_range=tuple(cfg.data.n_vars_range),
            leaf_prob=cfg.data.leaf_prob,
            steps_range=tuple(cfg.data.steps_range),
            distractor_prob=cfg.data.distractor_prob,
            max_distractors=cfg.data.max_distractors,
        )

    train_ds = make(cfg.data.train_size, cfg.data.train_seed)
    val_ds = make(cfg.data.val_size, cfg.data.val_seed)
    sampler = FreshEpochSampler(train_ds, seed=cfg.seed)
    collate = partial(collate_lm, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        num_workers=2,
        collate_fn=collate,
    )
    model = TokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(cfg.device)
    opt = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    total_steps = cfg.train.epochs * len(train_loader)
    logger = MetricLogger(out_dir)
    step, best = 0, float("inf")
    for epoch in range(cfg.train.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            tokens = batch["tokens"].to(cfg.device)
            prompt_len = batch["prompt_len"].to(cfg.device)
            for group in opt.param_groups:
                group["lr"] = cfg.train.lr * cosine_warmup(
                    step, total_steps, cfg.train.warmup_steps
                )
            out = model(tokens, prompt_len)
            loss, items = losses(out, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            model.update_teacher(
                ema_momentum(
                    step,
                    total_steps,
                    cfg.train.ema_start,
                    cfg.train.ema_end,
                )
            )
            if step % cfg.train.log_every == 0:
                logger.log(step, items, prefix="train/")
            step += 1

        model.eval()
        sums: dict[str, float] = {}
        count = 0
        state_features, macro_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= cfg.train.eval_batches:
                    break
                out = model(
                    batch["tokens"].to(cfg.device),
                    batch["prompt_len"].to(cfg.device),
                )
                _, items = losses(out, cfg)
                for name, value in items.items():
                    sums[name] = sums.get(name, 0.0) + value
                state_features.append(
                    out["target"].reshape(-1, out["target"].shape[-1])[
                        out["valid"].reshape(-1)
                    ]
                )
                macro_features.append(
                    out["macro_codes"].reshape(
                        -1, out["macro_codes"].shape[-1]
                    )[out["high_valid"].reshape(-1)]
                )
                count += 1
        metrics = {name: value / count for name, value in sums.items()}
        sf = torch.cat(state_features)
        mf = torch.cat(macro_features)
        metrics.update(
            state_std=feature_std(sf),
            state_rank=effective_rank(sf[:4096]),
            macro_std=feature_std(mf),
            macro_rank=effective_rank(mf[:4096]),
        )
        logger.log(step, metrics, prefix="val/")
        payload = {
            "model": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(payload, out_dir / "last.pt")
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save(payload, out_dir / "best.pt")
        print(
            f"[epoch {epoch}] "
            + "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())),
            flush=True,
        )
    logger.close()


if __name__ == "__main__":
    main()
