"""Train the flat-backbone intent JEPA on faithful iGSM.

    .venv/bin/python scripts/train_flat_jepa.py train.max_steps=50 hydra.run.dir=...

Every config key is read explicitly; unread keys abort the run (this codebase
silently drops unwired hydra keys otherwise).
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial
import json
import math
from pathlib import Path
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab
from textjepa.data.flat_stream import FlatIntentStreamDataset, collate_flat
from textjepa.data.sampling import FreshEpochSampler
from textjepa.models.flat_intent_jepa import FlatIntentJEPA
from textjepa.objectives import (
    CompositeObjective,
    CounterfactualStatePrediction,
    EnergyCFFeasibilityRank,
    GeoAdvantageRegression,
    GeoHorizonRank,
    IntentPriorLM,
    LatentPrediction,
    ObservedActionLDAD,
    VICReg,
)
from textjepa.training.loggers import MetricLogger
from textjepa.training.optim import cosine_warmup, ema_momentum
from textjepa.utils import seed_everything


class Tracked:
    """Attribute access wrapper recording every consumed config leaf."""

    def __init__(self, data: dict, prefix: str, seen: set):
        self._d, self._p, self._seen = data, prefix, seen

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        if key not in self._d:
            raise KeyError(f"missing config key {self._p}{key}")
        value = self._d[key]
        if isinstance(value, dict):
            return Tracked(value, f"{self._p}{key}.", self._seen)
        self._seen.add(f"{self._p}{key}")
        return value

    def get(self, key, default=None):
        if key in self._d:
            return getattr(self, key)
        return default

    def as_dict(self) -> dict:
        for k in self._d:
            getattr(self, k)
        return self._d


def _leaves(d: dict, prefix: str = "") -> set:
    out = set()
    for k, v in d.items():
        if isinstance(v, dict):
            out |= _leaves(v, f"{prefix}{k}.")
        else:
            out.add(f"{prefix}{k}")
    return out


def build_objective(oc) -> CompositeObjective:
    objs = {
        "latent_pred": LatentPrediction(oc.latent_pred.kind, oc.latent_pred.norm_targets),
        "vicreg": VICReg(oc.vicreg.std_target, oc.vicreg.cov_weight, oc.vicreg.action_weight),
        "counterfactual_state": CounterfactualStatePrediction(
            oc.counterfactual_state.kind, oc.counterfactual_state.norm_targets
        ),
        "geo_horizon_rank": GeoHorizonRank(
            margin=oc.geo_horizon_rank.margin, label_gap=oc.geo_horizon_rank.label_gap,
            kind=oc.geo_horizon_rank.kind, temperature=oc.geo_horizon_rank.temperature,
            teacher_temperature=oc.geo_horizon_rank.teacher_temperature,
        ),
        "geo_advantage_mse": GeoAdvantageRegression(oc.geo_advantage_mse.target_scale),
        "observed_action_ldad": ObservedActionLDAD(),
        "energy_cf_feasibility_rank": EnergyCFFeasibilityRank(),
        "intent_prior_lm": IntentPriorLM(),
    }
    weights = {name: float(getattr(oc, name).weight) for name in objs}
    return CompositeObjective(objs, weights)


def build_data(dc, vocab, split: str, lm_loss_on: str, size=None):
    base = FaithfulDataset(
        vocab,
        size=size if size is not None else (dc.train_size if split == "train" else dc.val_size),
        seed=dc.train_seed if split == "train" else dc.val_seed,
        max_op=dc.max_op, max_edge=dc.max_edge, op_range=tuple(dc.op_range),
        distractor_prob=dc.distractor_prob, max_distractors=dc.max_distractors,
        geo_rank_k=dc.geo_rank_k, geo_rank_horizon=max(dc.geo_rank_horizons),
        geo_rank_horizons=list(dc.geo_rank_horizons), geo_rank_rollout_for_h1=True,
        geo_rank_rollouts=dc.geo_rank_rollouts,
        invalid_counterfactual_k=dc.invalid_counterfactual_k,
        invalid_counterfactual_unresolved_only=dc.invalid_counterfactual_unresolved_only,
        invalid_counterfactual_resolved_k=dc.invalid_counterfactual_resolved_k,
        rollout_counterfactual_k=dc.rollout_counterfactual_k,
        all_action_supervision=True,
        necessary_range=tuple(dc.necessary_range) if dc.necessary_range else (None, None),
    )
    return FlatIntentStreamDataset(base, lm_loss_on=lm_loss_on)


MODULE_GROUPS = ("encoder", "predictor", "horizon_energy_head",
                 "observed_action_decoder")


def grad_norm_report(model, objective, out, batch) -> dict:
    """Per-term, per-module gradient norms on one batch (diagnostic)."""
    report = {}
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for name, obj in objective.objectives.items():
        w = objective.weights.get(name, 1.0)
        if w == 0.0:
            continue
        loss = obj(out, batch) * w
        grads = torch.autograd.grad(
            loss, [p for _, p in params], retain_graph=True, allow_unused=True,
        )
        norms = defaultdict(float)
        for (pname, _), g in zip(params, grads):
            if g is None:
                continue
            group = next((m for m in MODULE_GROUPS if pname.startswith(m + ".")), "other")
            if pname.startswith("encoder.tok.") or pname.startswith("encoder.head."):
                group = "lm_head/embedding"
            norms[group] += float(g.float().pow(2).sum())
        report[name] = {k: math.sqrt(v) for k, v in norms.items()}
    return report


@hydra.main(config_path="../configs", config_name="flat_jepa", version_base="1.3")
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=True)
    seen: set = set()
    c = Tracked(raw, "", seen)
    seed_everything(c.seed)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    device = torch.device(c.device)
    _ = c.run_name
    vocab = cached_faithful_vocab(c.data.vocab_max_op, c.data.vocab_max_edge)
    model = FlatIntentJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **c.model.as_dict()
    ).to(device)
    objective = build_objective(c.objective)
    lm_loss_on = c.model.lm_loss_on
    train_ds = build_data(c.data, vocab, "train", lm_loss_on)
    val_ds = build_data(c.data, vocab, "val", lm_loss_on)
    coll = partial(collate_flat, pad_id=vocab.pad_id)
    tc = c.train
    micro = int(tc.microbatch_size)
    if tc.batch_size % micro:
        raise ValueError("microbatch_size must divide batch_size")
    accum = tc.batch_size // micro
    sampler = FreshEpochSampler(train_ds, seed=c.seed)
    train_loader = DataLoader(
        train_ds, batch_size=micro, sampler=sampler, num_workers=tc.num_workers,
        collate_fn=coll, drop_last=True, persistent_workers=tc.num_workers > 0,
        prefetch_factor=4 if tc.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=micro, num_workers=min(tc.num_workers, 4), collate_fn=coll,
    )
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and not n.startswith("teacher.") and p.requires_grad]

    def groups(params, lr):
        decay = [p for p in params if p.dim() >= 2]
        nodecay = [p for p in params if p.dim() < 2]
        return [
            {"params": decay, "weight_decay": tc.weight_decay, "base_lr": lr},
            {"params": nodecay, "weight_decay": 0.0, "base_lr": lr},
        ]

    param_groups = groups(rest, tc.lr)
    enc_lr = tc.lr * tc.encoder_lr_mult
    if enc_params:
        param_groups += groups(enc_params, enc_lr)
    opt = torch.optim.AdamW(param_groups, lr=tc.lr, betas=tuple(tc.betas), fused=device.type == "cuda")
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n_total/1e6:.1f}M total, {n_train/1e6:.1f}M trainable (encoder_mode={model.encoder_mode})")
    precision = tc.precision
    if precision not in {"fp32", "bf16"}:
        raise ValueError(precision)
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if precision == "bf16" and device.type == "cuda" else (lambda: torch.autocast("cpu", enabled=False))
    )
    steps_per_epoch = len(train_loader) // accum
    total = tc.epochs * steps_per_epoch
    max_steps = tc.max_steps
    log_every, eval_batches = tc.log_every, tc.eval_batches
    grad_report_steps = int(tc.grad_norm_report_steps)
    ckpt_every = int(tc.checkpoint_every_steps)
    lr_floor, warmup, grad_clip = tc.lr_floor, tc.warmup_steps, tc.grad_clip
    ema_range = (tc.ema_start, tc.ema_end)
    epochs = tc.epochs
    unread = _leaves(raw) - seen - {"hydra.run.dir", "hydra.output_subdir", "hydra.job.chdir"}
    unread = {u for u in unread if not u.startswith("hydra.")}
    if unread:
        raise RuntimeError(f"unwired config keys: {sorted(unread)}")
    logger = MetricLogger(out_dir)
    model_dir = out_dir

    def to_dev(b):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

    def save(tag, epoch, step, extra=None):
        ckpt = {
            "model": model.state_dict(), "cfg": raw, "epoch": epoch, "step": step,
            "vocab_caps": [c.data.vocab_max_op, c.data.vocab_max_edge],
            "kind": "flat_intent_jepa", **(extra or {}),
        }
        torch.save(ckpt, model_dir / f"{tag}.pt")

    step, best = 0, float("inf")
    done = False
    t_start = time.time()
    problems_seen = 0
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        micro_i = 0
        t_epoch = time.time()
        acc_items = defaultdict(float)
        for batch in train_loader:
            batch = to_dev(batch)
            batch["pad_id"] = vocab.pad_id
            with autocast():
                out = model(batch)
                loss, items = objective(out, batch)
            if step < grad_report_steps and micro_i % accum == 0:
                report = grad_norm_report(model, objective, out, batch)
                print(f"[grad-norm report step {step}]")
                for name, groups_ in report.items():
                    print(f"  {name:28s} " + "  ".join(f"{k}={v:.3e}" for k, v in sorted(groups_.items())))
                (out_dir / f"grad_norms_step{step}.json").write_text(json.dumps(report, indent=2) + "\n")
            (loss / accum).backward()
            for k, v in items.items():
                acc_items[k] += v / accum
            for k in ("energy_cf_depth_acc", "energy_cf_depth_pairs"):
                if k in out.extras:
                    acc_items[k] += float(out.extras[k]) / accum
            micro_i += 1
            problems_seen += batch["tokens"].shape[0]
            if micro_i % accum:
                continue
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * cosine_warmup(step, total, warmup, lr_floor)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            model.update_teachers(ema_momentum(step, total, *ema_range))
            if step % log_every == 0:
                elapsed = time.time() - t_start
                metrics = {**acc_items, "total": float(loss.item()), "grad_norm": float(gn),
                           "lr": opt.param_groups[0]["lr"],
                           "s_per_problem": elapsed / max(problems_seen, 1),
                           "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0}
                logger.log(step, metrics, prefix="train/")
                print(f"[ep {epoch} step {step}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
            acc_items = defaultdict(float)
            step += 1
            if ckpt_every and step % ckpt_every == 0:
                save("last", epoch, step)
            if max_steps is not None and step >= max_steps:
                done = True
                break
        # ---- validation
        model.eval()
        vals = defaultdict(float)
        n_val = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= eval_batches:
                    break
                batch = to_dev(batch)
                batch["pad_id"] = vocab.pad_id
                with autocast():
                    out = model(batch)
                    loss, items = objective(out, batch)
                for k, v in items.items():
                    vals[k] += v
                vals["total"] += float(loss.item())
                n_val += 1
        vals = {k: v / max(n_val, 1) for k, v in vals.items()}
        logger.log(step, vals, prefix="val/")
        print(f"[epoch {epoch}] val " + "  ".join(f"{k}={v:.4f}" for k, v in vals.items())
              + f"  epoch_time_s={time.time()-t_epoch:.0f}", flush=True)
        save("last", epoch, step, {"val": vals})
        if vals["total"] < best:
            best = vals["total"]
            save("best", epoch, step, {"val": vals})
        if done:
            break
    (out_dir / "training_complete.json").write_text(json.dumps({
        "status": "completed", "steps": step, "best_val_total": best,
        "s_per_problem": (time.time() - t_start) / max(problems_seen, 1),
    }, indent=2) + "\n")
    logger.close()


if __name__ == "__main__":
    main()
