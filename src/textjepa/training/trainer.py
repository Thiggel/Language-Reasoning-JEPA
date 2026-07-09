"""Generic JEPA trainer: works for any model exposing forward(batch)->outputs
and update_teachers(momentum), with a CompositeObjective."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from textjepa.probing.probes import ridge_probe_accuracy
from textjepa.training.loggers import MetricLogger
from textjepa.training.optim import build_optimizer, cosine_warmup, ema_momentum
from textjepa.utils.metrics import effective_rank, feature_std


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


class Trainer:
    def __init__(self, cfg, model, objective, train_loader, val_loader, out_dir):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.objective = objective.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_dir = Path(out_dir)
        self.logger = MetricLogger(self.out_dir)

        tc = cfg.train
        self.epochs = tc.epochs
        self.total_steps = tc.epochs * len(train_loader)
        self.opt = build_optimizer(model, tc.lr, tc.weight_decay)
        self.clip = tc.grad_clip
        self.warmup = tc.warmup_steps
        self.ema_range = (tc.ema_start, tc.ema_end)
        self.log_every = tc.log_every
        self.eval_batches = tc.eval_batches
        self.step = 0

    def fit(self) -> dict[str, float]:
        best = float("inf")
        val_metrics: dict[str, float] = {}
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            val_metrics = self.evaluate()
            self.logger.log(self.step, val_metrics, prefix="val/")
            self._checkpoint("last.pt", epoch, val_metrics)
            if val_metrics["loss"] < best:
                best = val_metrics["loss"]
                self._checkpoint("best.pt", epoch, val_metrics)
            summary = "  ".join(f"{k}={v:.4f}" for k, v in sorted(val_metrics.items()))
            print(f"[epoch {epoch}] {summary}", flush=True)
        self.logger.close()
        return val_metrics

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        t0 = time.time()
        for batch in self.train_loader:
            batch = to_device(batch, self.device)
            lr_scale = cosine_warmup(self.step, self.total_steps, self.warmup)
            for g in self.opt.param_groups:
                g["lr"] = self.cfg.train.lr * lr_scale
            out = self.model(batch)
            loss, items = self.objective(out, batch)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.opt.step()
            self.model.update_teachers(
                ema_momentum(self.step, self.total_steps, *self.ema_range)
            )
            if self.step % self.log_every == 0:
                items.update(
                    loss=loss.item(),
                    lr=self.opt.param_groups[0]["lr"],
                    grad_norm=gnorm.item(),
                    steps_per_s=self.log_every / max(time.time() - t0, 1e-6),
                )
                t0 = time.time()
                self.logger.log(self.step, items, prefix="train/")
            self.step += 1

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        sums: dict[str, float] = {}
        n = 0
        feats: dict[str, list[torch.Tensor]] = {k: [] for k in (
            "state", "pred", "rollout", "delta", "value", "value_tgt",
            "step_value", "op", "necessary",
        )}
        for i, batch in enumerate(self.val_loader):
            if i >= self.eval_batches:
                break
            batch = to_device(batch, self.device)
            out = self.model(batch)
            loss, items = self.objective(out, batch)
            items["loss"] = loss.item()
            for k, v in items.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
            m = out.step_mask.reshape(-1)
            flat = lambda x: x.reshape(-1, x.shape[-1])[m]
            feats["state"].append(flat(out.step_states))
            feats["pred"].append(flat(out.preds))
            feats["rollout"].append(flat(out.rollout))
            feats["delta"].append(flat(out.step_states - out.prev_states))
            if "chunk_pred" in out.extras:
                feats.setdefault("chunkpred", []).append(flat(out.extras["chunk_pred"]))
            feats["step_value"].append(batch["value"].reshape(-1)[m])
            feats["op"].append(batch["op"].reshape(-1)[m])
            feats["necessary"].append(batch["necessary"].reshape(-1)[m])
            vm = torch.cat(
                [torch.ones_like(out.step_mask[:, :1]), out.step_mask], 1
            ).reshape(-1)
            feats["value"].append(out.value_pred.reshape(-1)[vm])
            rem = torch.cat(
                [batch["n_necessary"].unsqueeze(1), batch["remaining"]], 1
            ).reshape(-1)[vm]
            feats["value_tgt"].append(rem.float())

        metrics = {k: v / max(n, 1) for k, v in sums.items()}
        cat = {k: torch.cat(v) for k, v in feats.items() if v}
        metrics["state_std"] = feature_std(cat["state"])
        metrics["state_effrank"] = effective_rank(cat["state"][:4096])
        metrics["value_mae"] = (cat["value"] - cat["value_tgt"]).abs().mean().item()
        modulus = int(cat["step_value"].max().item()) + 1
        for src in ("state", "pred", "rollout", "chunkpred"):
            if src in cat:
                metrics[f"probe_value_{src}"] = ridge_probe_accuracy(
                    cat[src], cat["step_value"], modulus
                )
        metrics["probe_op_delta"] = ridge_probe_accuracy(cat["delta"], cat["op"], 4)
        metrics["probe_necessary_delta"] = ridge_probe_accuracy(
            cat["delta"], cat["necessary"], 2
        )
        return metrics

    def _checkpoint(self, name: str, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "cfg": OmegaConf.to_container(self.cfg, resolve=True),
                "epoch": epoch,
                "step": self.step,
                "metrics": metrics,
            },
            self.out_dir / name,
        )
