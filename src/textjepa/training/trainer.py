"""Generic JEPA trainer: works for any model exposing forward(batch)->outputs
and update_teachers(momentum), with a CompositeObjective."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from textjepa.objectives.geometry import goal_distances, velocity_cosines
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
        self.total_steps = tc.epochs * len(train_loader) // self.grad_accum_steps
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
        dataset = self.train_loader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)
        t0 = time.time()
        updates_since_log = 0
        self.opt.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(self.train_loader):
            batch = to_device(batch, self.device)
            lr_scale = cosine_warmup(self.step, self.total_steps, self.warmup)
            for g in self.opt.param_groups:
                g["lr"] = self.cfg.train.lr * lr_scale
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
                )
                t0 = time.time()
                updates_since_log = 0
                self.logger.log(self.step, items, prefix="train/")
            self.step += 1

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
            if prior_position is not None:
                steps = prior_position.shape[1]
                valid = out.step_mask[:, :steps] & batch["op"][:, :steps].eq(2)
                items["refinement_position_accuracy"] = (
                    prior_position.argmax(-1)[valid]
                    .eq(batch["edit_position"][:, :steps][valid])
                    .float().mean().item()
                )
                prior_content = out.extras["refinement_content_logits"]
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
