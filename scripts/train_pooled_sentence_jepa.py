"""Train the causal pooled-prefix token-action JEPA."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    from train_sentence_hierarchy import (
        _pairwise_advantage_loss, dense_loss, make_dataset, normalized_mse, vicreg,
    )
except ModuleNotFoundError:
    from scripts.train_sentence_hierarchy import (
        _pairwise_advantage_loss, dense_loss, make_dataset, normalized_mse, vicreg,
    )
from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.sampling import DistributedFreshEpochSampler, FreshEpochSampler
from textjepa.data.semantic_lm import collate_semantic_lm
from textjepa.models.pooled_sentence_jepa import PooledSentenceJEPA
from textjepa.training.loggers import MetricLogger
from textjepa.training.loading import performance_loader_kwargs
from textjepa.training.optim import build_optimizer, cosine_warmup, ema_momentum
from textjepa.training.distributed import barrier, close, initialize, wrap
from textjepa.training.scale import (
    crossed_milestones, exposure_milestones, save_checkpoint,
)
from textjepa.utils import seed_everything
from textjepa.utils.metrics import effective_rank, feature_std


def optimizer_step_count(micro_batches: int, accumulation: int) -> int:
    accumulation = max(1, int(accumulation))
    return (int(micro_batches) + accumulation - 1) // accumulation


def accumulation_group_size(
    micro_batch_index: int, micro_batches: int, accumulation: int
) -> int:
    """Actual divisor, including a possibly shorter final group."""
    accumulation = max(1, int(accumulation))
    group_start = (int(micro_batch_index) // accumulation) * accumulation
    return min(accumulation, int(micro_batches) - group_start)


def forward(model, batch, device):
    return model(
        batch["tokens"].to(device, non_blocking=True),
        batch["prompt_len"].to(device, non_blocking=True),
        batch["sentence_ends"].to(device, non_blocking=True),
    )


def _masked_sequence_ce(logits, targets, valid):
    losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return (losses * valid).sum(1) / valid.sum(1).clamp_min(1)


def _global_valid_features(features):
    """Autograd-safe variable-length gather for cross-rank VICReg."""
    if not torch.distributed.is_initialized():
        return features
    import torch.distributed.nn.functional as dist_f
    count = torch.tensor([len(features)], device=features.device, dtype=torch.long)
    counts = [torch.zeros_like(count) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(counts, count)
    maximum = max(int(item.item()) for item in counts)
    padded = F.pad(features, (0, 0, 0, maximum - len(features)))
    gathered = dist_f.all_gather(padded)
    return torch.cat([
        tensor[:int(size.item())] for tensor, size in zip(gathered, counts)
    ])


def compute_losses(out, cfg, model, batch, distributed_regularizer=False):
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
    regularizer_features = out["prev"][out["valid"]]
    if distributed_regularizer:
        regularizer_features = _global_valid_features(regularizer_features)
    regularizer = vicreg(regularizer_features, obj.covariance)
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
    decoder_ce = prediction.sum() * 0.0
    decoder_state_use = decoder_ce
    decoder_state_gap = decoder_ce.detach()
    if model.prefix_decoder is not None:
        decoded = model.prefix_decoder_batch(
            out, batch["tokens"].to(out["states"].device),
            batch["prompt_len"].to(out["states"].device),
            batch["sentence_ends"].to(out["states"].device),
        )
        correct = _masked_sequence_ce(
            decoded["logits"], decoded["targets"], decoded["valid"]
        )
        shuffled = _masked_sequence_ce(
            decoded["shuffled_logits"], decoded["targets"], decoded["valid"]
        )
        decoder_ce = correct.mean()
        decoder_state_use = F.relu(
            float(obj.decoder_state_margin) + correct - shuffled
        ).mean()
        decoder_state_gap = (shuffled - correct).mean()
    total = (
        obj.prediction * prediction + obj.dense * dense
        + obj.token_prior * prior + obj.goal * goal
        + obj.vicreg * regularizer + obj.gar * gar_total
        + obj.prefix_decoder * decoder_ce
        + obj.decoder_state_use * decoder_state_use
    )
    items = {
        "prediction": prediction, "dense": dense, "token_prior": prior,
        "token_prior_accuracy": prior_accuracy, "goal": goal,
        "vicreg": regularizer, "gar_regression": gar_regression,
        "gar_ranking": gar_ranking, "gar_counterfactual_mse": gar_dynamics,
        "gar_advantage_std": cf["advantage_target"].std(),
        "prefix_decoder": decoder_ce, "decoder_state_use": decoder_state_use,
        "decoder_state_gap": decoder_state_gap,
        # Select the planning checkpoint using deployable dynamics, proposal,
        # and value objectives. Auxiliary reconstruction/regularization terms
        # must not make decoder and no-decoder checkpoint selection inequivalent.
        "selection": (
            obj.prediction * prediction + obj.dense * dense
            + obj.token_prior * prior + obj.gar * gar_total
        ).detach(),
    }
    for horizon, (rollout, target, mask) in enumerate(zip(
        out["dense_predictions"], out["dense_targets"], out["dense_masks"]
    ), start=1):
        weight = mask.float()
        denominator = weight.sum().clamp_min(1)
        raw = (rollout - target).square().mean(-1)
        cosine = 1.0 - F.cosine_similarity(rollout, target, dim=-1)
        items[f"rollout_normalized_mse_h{horizon}"] = normalized_mse(
            rollout, target, mask
        ).detach()
        items[f"rollout_raw_mse_h{horizon}"] = (
            raw * weight
        ).sum().div(denominator).detach()
        items[f"rollout_cosine_distance_h{horizon}"] = (
            cosine * weight
        ).sum().div(denominator).detach()
    return total, items


@torch.no_grad()
def dependence_diagnostics(model, out):
    valid = out["valid"]
    shuffled_state = model.predictor(out["prev"].roll(1, 0), out["actions"], valid)
    zero_action = model.predictor(out["prev"], torch.zeros_like(out["actions"]), valid)
    shuffled_action = model.predictor(out["prev"], out["actions"].roll(1, 0), valid)
    return {
        "prediction_shuffled_state": normalized_mse(shuffled_state, out["target"], valid),
        "prediction_zero_action": normalized_mse(zero_action, out["target"], valid),
        "prediction_shuffled_action": normalized_mse(shuffled_action, out["target"], valid),
    }


@hydra.main(config_path="../configs", config_name="pooled_sentence_jepa", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    distributed = initialize(cfg.device)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    if distributed.primary:
        print(OmegaConf.to_yaml(cfg))
    vocab = build_vocab(cfg.data.modulus)
    train_ds = make_dataset(cfg, vocab, cfg.data.train_size, cfg.data.train_seed)
    val_ds = make_dataset(cfg, vocab, cfg.data.val_size, cfg.data.val_seed)
    sampler = (
        DistributedFreshEpochSampler(
            train_ds, distributed.rank, distributed.world_size, seed=cfg.seed
        ) if distributed.world_size > 1 else FreshEpochSampler(train_ds, seed=cfg.seed)
    )
    collate = partial(collate_semantic_lm, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
        num_workers=cfg.train.num_workers, collate_fn=collate, drop_last=True,
        **performance_loader_kwargs(cfg.train.num_workers, distributed.device),
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.train.get("eval_batch_size", cfg.train.batch_size)),
        num_workers=cfg.train.num_workers, collate_fn=collate,
        **performance_loader_kwargs(
            cfg.train.num_workers, distributed.device, persistent=False,
        ),
    )
    raw_model = PooledSentenceJEPA(
        len(vocab), vocab.pad_id, period_id=vocab.token_to_id["."],
        question_id=vocab.token_to_id["?"], **cfg.model,
    ).to(distributed.device)
    if distributed.primary:
        print({
            "trainable_parameters": sum(p.numel() for p in raw_model.parameters() if p.requires_grad),
            "ema_target_parameters": sum(p.numel() for p in raw_model.teacher.parameters()),
        }, flush=True)
    model = wrap(raw_model, distributed)
    optimizer = build_optimizer(
        raw_model, cfg.train.lr, cfg.train.weight_decay,
        tuple(cfg.train.get("betas", (0.9, 0.95))),
    )
    accumulation = max(1, int(cfg.train.get("gradient_accumulation_steps", 1)))
    total_steps = cfg.train.epochs * optimizer_step_count(
        len(train_loader), accumulation
    )
    effective_batch = int(cfg.train.batch_size) * accumulation * distributed.world_size
    milestones = exposure_milestones(cfg.train.get("checkpoint_examples", []))
    logger = MetricLogger(out_dir) if distributed.primary else None
    step, best, first_epoch = 0, float("inf"), 0
    resume = cfg.train.get("resume_from")
    if resume:
        state = torch.load(str(resume), map_location=distributed.device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, best = int(state["step"]), float(state["best"])
        first_epoch = int(state["epoch"]) + 1
    if distributed.primary:
        print({
            "world_size": distributed.world_size,
            "micro_batch_size_per_gpu": int(cfg.train.batch_size),
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": effective_batch,
            "vicreg_batch_size": int(cfg.train.batch_size) * distributed.world_size,
            "optimizer_steps": total_steps,
            "sequence_presentations": int(cfg.data.train_size) * int(cfg.train.epochs),
        }, flush=True)
    use_bf16 = str(cfg.train.get("precision", "fp32")).lower() == "bf16"
    for epoch in range(first_epoch, cfg.train.epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for micro_index, batch in enumerate(train_loader):
            if micro_index % accumulation == 0:
                for group in optimizer.param_groups:
                    group["lr"] = cfg.train.lr * cosine_warmup(
                        step, total_steps, cfg.train.warmup_steps
                    )
            should_step = (
                (micro_index + 1) % accumulation == 0
                or micro_index + 1 == len(train_loader)
            )
            sync = nullcontext() if should_step or distributed.world_size == 1 else model.no_sync()
            with sync:
                with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
                    out = forward(model, batch, distributed.device)
                    loss, items = compute_losses(
                        out, cfg, raw_model, batch,
                        distributed_regularizer=distributed.world_size > 1,
                    )
                divisor = accumulation_group_size(
                    micro_index, len(train_loader), accumulation
                )
                (loss / divisor).backward()
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            raw_model.update_teacher(ema_momentum(
                step, total_steps, cfg.train.ema_start, cfg.train.ema_end
            ))
            if distributed.primary and step % cfg.train.log_every == 0:
                logger.log(step, {k: float(v.detach()) for k, v in items.items()}, prefix="train/")
            step += 1
        barrier(distributed)
        if distributed.primary:
            raw_model.eval()
            sums, count, features = {}, 0, []
            with torch.no_grad():
                for index, batch in enumerate(val_loader):
                    if index >= cfg.train.eval_batches:
                        break
                    out = forward(raw_model, batch, distributed.device)
                    _, items = compute_losses(out, cfg, raw_model, batch)
                    if index == 0:
                        items.update(dependence_diagnostics(raw_model, out))
                    for key, value in items.items():
                        sums[key] = sums.get(key, 0.0) + float(value)
                    features.append(out["target"][out["valid"]])
                    count += 1
            metrics = {
                key: value / (1 if key.startswith("prediction_") and key != "prediction" else count)
                for key, value in sums.items()
            }
            feature = torch.cat(features)
            metrics.update(state_std=feature_std(feature), state_rank=effective_rank(feature[:4096]))
            logger.log(step, metrics, prefix="val/")
            previous = epoch * int(cfg.data.train_size)
            exposure = (epoch + 1) * int(cfg.data.train_size)
            improved = metrics["selection"] < best
            best = min(best, metrics["selection"])
            payload = {
                "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "epoch": epoch, "step": step, "best": best,
                "examples_seen": exposure, "metrics": metrics,
            }
            torch.save(payload, out_dir / "last.pt")
            if improved:
                torch.save(payload, out_dir / "best.pt")
            for _ in crossed_milestones(previous, exposure, milestones):
                save_checkpoint(payload, out_dir, exposure)
            print(f"[epoch {epoch}] examples={exposure} " + "  ".join(
                f"{key}={value:.4f}" for key, value in sorted(metrics.items())
            ), flush=True)
        barrier(distributed)
    if logger is not None:
        logger.close()
    close(distributed)


if __name__ == "__main__":
    main()
