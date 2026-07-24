"""Generic JEPA trainer: works for any model exposing forward(batch)->outputs
and update_teachers(momentum), with a CompositeObjective."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import OmegaConf

from textjepa.objectives.geometry import goal_distances, velocity_cosines
from textjepa.models.layers import attention_backend_summary
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
        effective_batch = int(tc.batch_size)
        microbatch = int(tc.get("microbatch_size", effective_batch))
        if effective_batch % microbatch:
            raise ValueError("microbatch_size must divide batch_size")
        self.grad_accum_steps = effective_batch // microbatch
        if len(train_loader) % self.grad_accum_steps:
            raise ValueError("loader microbatches must form complete effective batches")
        inferred_steps = tc.epochs * len(train_loader) // self.grad_accum_steps
        self.total_steps = int(tc.get("max_steps", inferred_steps))
        self.opt = build_optimizer(
            model, tc.lr, tc.weight_decay,
            betas=tuple(tc.get("betas", (0.9, 0.95))),
        )
        self.clip = tc.grad_clip
        self.warmup = tc.warmup_steps
        self.lr_floor = float(tc.get("lr_floor", 0.05))
        self.ema_range = (tc.ema_start, tc.ema_end)
        self.log_every = tc.log_every
        self.eval_batches = tc.eval_batches
        self.precision = str(tc.get("precision", "fp32"))
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError(f"unsupported training precision: {self.precision}")
        self.step = 0
        self.best = float("inf")
        self.last_val_metrics: dict[str, float] = {}
        self.start_epoch = 0
        self.start_micro_step = 0
        self._epoch_micro_step = 0
        self.stop_after_steps = tc.get("stop_after_steps")
        if self.stop_after_steps is not None:
            self.stop_after_steps = int(self.stop_after_steps)
            if not 0 < self.stop_after_steps <= self.total_steps:
                raise ValueError("stop_after_steps must lie in [1, max_steps]")
        resume = tc.get("resume_from")
        if resume:
            payload = torch.load(Path(resume), map_location=self.device,
                                 weights_only=False)
            self.model.load_state_dict(payload["model"])
            self.opt.load_state_dict(payload["optimizer"])
            self.step = int(payload["step"])
            self.best = float(payload.get("best", self.best))
            self.last_val_metrics = dict(payload.get("metrics", {}))
            self.start_epoch = int(payload["epoch"])
            self.start_micro_step = int(payload.get("epoch_micro_step", 0))
            if payload.get("epoch_complete", False):
                self.start_epoch += 1
                self.start_micro_step = 0
            if self.start_micro_step % self.grad_accum_steps:
                raise ValueError("resume checkpoint is not on an optimizer boundary")
            print(
                f"resumed step={self.step} epoch={self.start_epoch} "
                f"micro_step={self.start_micro_step} from {resume}",
                flush=True,
            )
        self._reported_attention_backend = False

    def _autocast(self):
        if self.precision == "bf16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def fit(self) -> dict[str, float]:
        val_metrics: dict[str, float] = {}
        for epoch in range(self.start_epoch, self.epochs):
            start_micro_step = self.start_micro_step if epoch == self.start_epoch else 0
            epoch_complete = self._train_epoch(epoch, start_micro_step)
            val_metrics = self.evaluate()
            self.logger.log(self.step, val_metrics, prefix="val/")
            self.last_val_metrics = val_metrics
            self._checkpoint("last.pt", epoch, val_metrics, epoch_complete)
            if val_metrics["loss"] < self.best:
                self.best = val_metrics["loss"]
                self._checkpoint("best.pt", epoch, val_metrics, epoch_complete)
            summary = "  ".join(f"{k}={v:.4f}" for k, v in sorted(val_metrics.items()))
            print(f"[epoch {epoch}] {summary}", flush=True)
            if self.step >= self.total_steps or (
                self.stop_after_steps is not None
                and self.step >= self.stop_after_steps
            ):
                break
        self.logger.close()
        return val_metrics

    def _train_epoch(self, epoch: int, start_micro_step: int = 0) -> bool:
        self.model.train()
        dataset = self.train_loader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)
        for sampler in (self.train_loader.sampler, self.train_loader.batch_sampler):
            set_start = getattr(sampler, "set_start", None)
            if set_start is not None:
                # Ordinary DataLoader samplers yield individual indices, while
                # grouped batch samplers yield already formed microbatches.
                offset = start_micro_step
                if sampler is self.train_loader.sampler:
                    offset *= int(self.cfg.train.get(
                        "microbatch_size", self.cfg.train.batch_size
                    ))
                set_start(offset)
        t0 = time.time()
        updates_since_log = 0
        self.opt.zero_grad(set_to_none=True)
        epoch_complete = True
        for local_micro_step, batch in enumerate(self.train_loader):
            if self.step >= self.total_steps or (
                self.stop_after_steps is not None
                and self.step >= self.stop_after_steps
            ):
                epoch_complete = False
                break
            micro_step = start_micro_step + local_micro_step
            self._epoch_micro_step = micro_step + 1
            batch = to_device(batch, self.device)
            lr_scale = cosine_warmup(
                self.step, self.total_steps, self.warmup, self.lr_floor
            )
            for g in self.opt.param_groups:
                g["lr"] = self.cfg.train.lr * lr_scale
            with self._autocast():
                out = self.model(batch)
                loss, items = self.objective(out, batch)
            if not self._reported_attention_backend:
                print(
                    "attention_backends="
                    + ",".join(attention_backend_summary(self.model)),
                    flush=True,
                )
                self._reported_attention_backend = True
            support_logits = out.extras.get("action_support_logits")
            if support_logits is not None:
                support_valid = out.extras["action_support_valid"]
                support_target = out.extras["action_support_target"]
                support_pred = support_logits >= 0
                items["action_feasibility_accuracy"] = (
                    support_pred[support_valid]
                    .eq(support_target[support_valid])
                    .float().mean().item()
                )
                positive = support_valid & support_target
                negative = support_valid & ~support_target
                items["action_feasibility_positive_logit"] = (
                    support_logits[positive].mean().item()
                )
                items["action_feasibility_negative_logit"] = (
                    support_logits[negative].mean().item()
                )
            (loss / self.grad_accum_steps).backward()
            if (micro_step + 1) % self.grad_accum_steps:
                continue
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.model.update_teachers(
                ema_momentum(self.step, self.total_steps, *self.ema_range)
            )
            updates_since_log += 1
            if self.step % self.log_every == 0:
                items.update(
                    loss=loss.item(),
                    lr=self.opt.param_groups[0]["lr"],
                    grad_norm=gnorm.item(),
                    steps_per_s=updates_since_log / max(
                        time.time() - t0, 1e-6
                    ),
                    peak_memory_gib=(
                        torch.cuda.max_memory_allocated(self.device) / 2**30
                        if self.device.type == "cuda" else 0.0
                    ),
                )
                t0 = time.time()
                updates_since_log = 0
                self.logger.log(self.step, items, prefix="train/")
            self.step += 1
            checkpoint_every = int(self.cfg.train.get("checkpoint_every_steps", 0))
            if checkpoint_every and self.step % checkpoint_every == 0:
                self._checkpoint("last.pt", epoch, self.last_val_metrics, False)
        return epoch_complete

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        sums: dict[str, float] = {}
        n = 0
        feats: dict[str, list[torch.Tensor]] = {k: [] for k in (
            "state", "pred", "rollout", "delta", "value", "value_tgt",
            "step_value", "op", "necessary", "action",
        )}
        sentence_stream = False
        for i, batch in enumerate(self.val_loader):
            if i >= self.eval_batches:
                break
            batch = to_device(batch, self.device)
            with self._autocast():
                out = self.model(batch)
                loss, items = self.objective(out, batch)
            support_logits = out.extras.get("action_support_logits")
            if support_logits is not None:
                support_valid = out.extras["action_support_valid"]
                support_target = out.extras["action_support_target"]
                support_pred = support_logits >= 0
                items["action_feasibility_accuracy"] = (
                    support_pred[support_valid]
                    .eq(support_target[support_valid])
                    .float().mean().item()
                )
                positive = support_valid & support_target
                negative = support_valid & ~support_target
                items["action_feasibility_positive_logit"] = (
                    support_logits[positive].mean().item()
                )
                items["action_feasibility_negative_logit"] = (
                    support_logits[negative].mean().item()
                )
            logits = out.extras.get("observed_action_logits")
            if logits is not None:
                target = batch["action_tokens"]
                width = min(logits.shape[-2], target.shape[-1])
                prediction = logits[..., :width, :].argmax(-1)
                target = target[..., :width]
                token_mask = out.step_mask.unsqueeze(-1) & target.ne(0)
                correct = prediction.eq(target)
                items["observed_action_token_accuracy"] = (
                    correct[token_mask].float().mean().item()
                )
                sequence_correct = (correct | ~token_mask).all(-1)
                items["observed_action_sequence_exact"] = (
                    sequence_correct[out.step_mask].float().mean().item()
                )
            prior_position = out.extras.get("refinement_position_logits")
            prior_content = out.extras.get("refinement_content_logits")
            if prior_position is not None:
                steps = prior_position.shape[1]
                valid = out.step_mask[:, :steps] & batch["op"][:, :steps].eq(2)
                items["refinement_position_accuracy"] = (
                    prior_position.argmax(-1)[valid]
                    .eq(batch["edit_position"][:, :steps][valid])
                    .float().mean().item()
                )
            if prior_content is not None:
                steps = prior_content.shape[1]
                valid = out.step_mask[:, :steps] & batch["op"][:, :steps].eq(2)
                if prior_content.ndim == 4:
                    b, t, width, _ = prior_content.shape
                    row = torch.arange(b, device=prior_content.device)[:, None]
                    time = torch.arange(t, device=prior_content.device)[None, :]
                    slot = batch["edit_position"][:, :steps].clamp(0, width - 1)
                    prior_content = prior_content[row, time, slot]
                items["refinement_content_accuracy"] = (
                    prior_content.argmax(-1)[valid]
                    .eq(batch["edit_content_token"][:, :steps][valid])
                    .float().mean().item()
                )
            multistep_logits = out.extras.get("observed_action_multistep_logits")
            if multistep_logits is not None and multistep_logits.shape[1] > 0:
                horizon = multistep_logits.shape[-3]
                n_starts = multistep_logits.shape[1]
                target = torch.stack(
                    [batch["action_tokens"][:, j : j + n_starts]
                     for j in range(horizon)],
                    dim=2,
                )
                valid = torch.stack(
                    [out.step_mask[:, j : j + n_starts]
                     for j in range(horizon)],
                    dim=2,
                )
                width = min(multistep_logits.shape[-2], target.shape[-1])
                prediction = multistep_logits[..., :width, :].argmax(-1)
                target = target[..., :width]
                token_mask = valid.unsqueeze(-1) & target.ne(0)
                correct = prediction.eq(target)
                items["observed_action_token_accuracy"] = (
                    correct[token_mask].float().mean().item()
                )
                phrase_correct = (correct | ~token_mask).all(-1)
                items["observed_action_sequence_exact"] = (
                    phrase_correct[valid].float().mean().item()
                )
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
            feats["action"].append(flat(out.actions))
            if out.hi_mask is not None and "macro_codes" in out.extras:
                hm = out.hi_mask.reshape(-1)
                hflat = lambda x: x.reshape(-1, x.shape[-1])[hm]
                feats.setdefault("macro", []).append(
                    hflat(out.extras["macro_codes"])
                )
                feats.setdefault("hi_pred", []).append(hflat(out.hi_preds))
                feats.setdefault("hi_target", []).append(hflat(out.hi_targets))
                if "hi_value_pred" in out.extras:
                    feats.setdefault("hi_value", []).append(
                        out.extras["hi_value_pred"].reshape(-1)[hm]
                    )
                    feats.setdefault("hi_value_target", []).append(
                        out.extras["hi_value_target"].reshape(-1)[hm]
                    )
            if out.extras.get("sentence_stream", False):
                sentence_stream = True
                feats.setdefault("pred_std", []).append(
                    (0.5 * out.extras["pred_logvar"]).exp().reshape(
                        -1, out.preds.shape[-1]
                    )[m]
                )
                feats.setdefault("target_std", []).append(
                    (0.5 * out.extras["target_logvar"]).exp().reshape(
                        -1, out.preds.shape[-1]
                    )[m]
                )
                feats.setdefault("action_q_mu", []).append(
                    flat(out.extras["action_q_mu"])
                )
                feats.setdefault("action_p_mu", []).append(
                    flat(out.extras["action_p_mu"])
                )
                if "latent_ldad_pred" in out.extras:
                    feats.setdefault("ldad_error", []).append(
                        (out.extras["latent_ldad_pred"]
                         - out.extras["latent_ldad_tgt"]).pow(2).mean(-1)
                        .reshape(-1)[m]
                    )
                continue
            if "chunk_pred" in out.extras:
                feats.setdefault("chunkpred", []).append(flat(out.extras["chunk_pred"]))
            cos, cmask = velocity_cosines(out)
            feats.setdefault("traj_cos", []).append(cos.reshape(-1)[cmask.reshape(-1) > 0])
            gd = goal_distances(out)[:, 1:]
            feats.setdefault("goal_dist", []).append(gd.reshape(-1)[m])
            feats.setdefault("goal_dist_rem", []).append(
                batch["remaining"].reshape(-1)[m].float()
            )
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
        metrics["action_std"] = feature_std(cat["action"])
        metrics["action_effrank"] = effective_rank(cat["action"][:4096])
        if "macro" in cat:
            metrics["macro_std"] = feature_std(cat["macro"])
            metrics["macro_effrank"] = effective_rank(cat["macro"][:4096])
            hp = torch.nn.functional.layer_norm(
                cat["hi_pred"], cat["hi_pred"].shape[-1:]
            )
            ht = torch.nn.functional.layer_norm(
                cat["hi_target"], cat["hi_target"].shape[-1:]
            )
            metrics["hi_matched_l1"] = (hp - ht).abs().mean().item()
            if "hi_value" in cat:
                metrics["hi_value_mae"] = (
                    cat["hi_value"] - 5.0 * cat["hi_value_target"]
                ).abs().mean().item()
        if sentence_stream:
            metrics["pred_std"] = cat["pred_std"].mean().item()
            metrics["target_std"] = cat["target_std"].mean().item()
            metrics["action_q_mu_std"] = feature_std(cat["action_q_mu"])
            metrics["action_q_mu_effrank"] = effective_rank(
                cat["action_q_mu"][:4096]
            )
            metrics["action_p_mu_std"] = feature_std(cat["action_p_mu"])
            if "ldad_error" in cat:
                metrics["ldad_mse"] = cat["ldad_error"].mean().item()
            return metrics
        metrics["value_mae"] = (cat["value"] - cat["value_tgt"]).abs().mean().item()
        modulus = int(cat["step_value"].max().item()) + 1
        for src in ("state", "pred", "rollout", "chunkpred"):
            if src in cat:
                metrics[f"probe_value_{src}"] = ridge_probe_accuracy(
                    cat[src], cat["step_value"], modulus
                )
        metrics["traj_cos"] = cat["traj_cos"].mean().item()
        gd, rem = cat["goal_dist"], cat["goal_dist_rem"]
        gd, rem = gd - gd.mean(), rem - rem.mean()
        metrics["goal_dist_corr"] = (
            (gd * rem).sum() / (gd.norm() * rem.norm() + 1e-8)
        ).item()
        metrics["probe_op_delta"] = ridge_probe_accuracy(cat["delta"], cat["op"], 4)
        metrics["probe_necessary_delta"] = ridge_probe_accuracy(
            cat["delta"], cat["necessary"], 2
        )
        return metrics

    def _checkpoint(self, name: str, epoch: int, metrics: dict,
                    epoch_complete: bool) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.opt.state_dict(),
                "cfg": OmegaConf.to_container(self.cfg, resolve=True),
                "epoch": epoch,
                "step": self.step,
                "epoch_micro_step": self._epoch_micro_step,
                "epoch_complete": epoch_complete,
                "best": self.best,
                "metrics": metrics,
            },
            self.out_dir / name,
        )
