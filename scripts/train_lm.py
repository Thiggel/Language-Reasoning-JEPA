"""Train the decoder-only LM baseline (teacher-forced next-token CE).

    python scripts/train_lm.py +experiment=lm_9m
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import (
    FlattenedDiscourseLMDataset,
    IntentPolicyLMDataset,
    LMDataset,
    collate_lm,
)
from textjepa.data.sampling import DistributedFreshEpochSampler, FreshEpochSampler
from textjepa.models.lm_baseline import DecoderLM
from textjepa.training.loggers import MetricLogger
from textjepa.training.loading import performance_loader_kwargs
from textjepa.training.optim import build_optimizer, cosine_warmup
from textjepa.training.distributed import barrier, close, initialize, wrap
from textjepa.training.scale import (
    crossed_milestones, exposure_milestones, optimizer_steps, save_checkpoint,
)
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset


def rank_loss(model, batch, device, margin=1.0):
    rt = batch["rank_tokens"].to(device, non_blocking=True)
    rb = batch["rank_better"].to(device, non_blocking=True)
    rf = batch["rank_from"].to(device, non_blocking=True)
    B, K1, L = rt.shape
    lp = model.sequence_logprob(
        rt.reshape(B * K1, L), rf.repeat_interleave(K1)
    ).reshape(B, K1)
    diff = lp[:, 1:] - lp[:, :1]  # alt minus executed
    import torch.nn.functional as Fn
    loss = (rb == 1).float() * Fn.relu(margin + diff) + (
        (rb == -1).float() * Fn.relu(margin - diff)
    )
    n = (rb != 0).float().sum().clamp(min=1.0)
    return loss.sum() / n


def lm_loss(model, batch, device):
    base_model = model.module if hasattr(model, "module") else model
    tokens = batch["tokens"].to(device, non_blocking=True)
    logits = model(tokens)[:, :-1]
    tgt = tokens[:, 1:]
    if "loss_mask" in batch:
        # loss_mask marks target tokens in the unshifted input sequence.
        mask = batch["loss_mask"].to(device, non_blocking=True)[:, 1:] & (tgt != base_model.pad_id)
    else:
        pos = torch.arange(tgt.shape[1], device=device).unsqueeze(0)
        mask = (pos >= (batch["prompt_len"].to(device, non_blocking=True).unsqueeze(1) - 1)) & (
            tgt != base_model.pad_id
        )
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none"
    ).reshape(tgt.shape)
    return (ce * mask).sum() / mask.sum().clamp(min=1)


@hydra.main(config_path="../configs", config_name="lm", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    distributed = initialize(cfg.device)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    if distributed.primary:
        print(OmegaConf.to_yaml(cfg))
    device = distributed.device
    if cfg.data.get("name", "igsm") == "igsm_real":
        from textjepa.data.faithful import cached_faithful_vocab

        vocab = cached_faithful_vocab()
    else:
        vocab = build_vocab(cfg.data.modulus)

    def make(split):
        d = cfg.data
        target_kind = cfg.train.get("target_kind", "outcome")
        if target_kind == "intent":
            if cfg.train.get("rank_weight", 0):
                raise ValueError("ranking loss is not defined for the intent policy LM")
            return IntentPolicyLMDataset(build_dataset(cfg, vocab, split=split))
        if target_kind != "outcome":
            raise ValueError(f"unknown LM target_kind: {target_kind}")
        if d.get("name", "igsm") == "igsm_real":
            if cfg.train.get("rank_weight", 0):
                raise ValueError("faithful token-LM ranking is not implemented")
            return FlattenedDiscourseLMDataset(
                build_dataset(cfg, vocab, split=split)
            )
        size = d.val_size if split == "val" else d.train_size
        seed = d.val_seed if split == "val" else d.train_seed
        return LMDataset(
            vocab, size=size, seed=seed, n_alt=d.get("n_alt", 0),
            modulus=d.modulus,
            n_vars_range=tuple(d.n_vars_range), leaf_prob=d.leaf_prob,
            steps_range=tuple(d.steps_range), distractor_prob=d.distractor_prob,
            max_distractors=d.max_distractors,
        )

    coll = partial(collate_lm, pad_id=vocab.pad_id)
    train_ds = make("train")
    train_sampler = (
        DistributedFreshEpochSampler(
            train_ds, distributed.rank, distributed.world_size, seed=cfg.seed
        ) if distributed.world_size > 1 else FreshEpochSampler(train_ds, seed=cfg.seed)
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=train_sampler,
        num_workers=cfg.train.num_workers, collate_fn=coll, drop_last=True,
        **performance_loader_kwargs(cfg.train.num_workers, device),
    )
    val_loader = DataLoader(
        make("val"), batch_size=cfg.train.batch_size,
        num_workers=int(cfg.train.get("val_num_workers", cfg.train.num_workers)),
        collate_fn=coll,
        **performance_loader_kwargs(
            int(cfg.train.get("val_num_workers", cfg.train.num_workers)),
            device, persistent=False,
        ),
    )

    raw_model = DecoderLM(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
        ff_mult=cfg.model.ff_mult, max_len=cfg.model.max_len,
    ).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    if distributed.primary:
        print(f"LM parameters: {n_params / 1e6:.2f}M")
    model = wrap(raw_model, distributed)

    opt = build_optimizer(raw_model, cfg.train.lr, cfg.train.weight_decay,
                          tuple(cfg.train.get("betas", (0.9, 0.95))))
    accumulation = int(cfg.train.get("gradient_accumulation_steps", 1))
    steps_per_epoch = optimizer_steps(len(train_loader), accumulation)
    total = cfg.train.epochs * steps_per_epoch
    effective_batch = cfg.train.batch_size * distributed.world_size * accumulation
    milestones = exposure_milestones(cfg.train.get("checkpoint_examples", []))
    logger = MetricLogger(out_dir) if distributed.primary else None
    step, best, first_epoch = 0, float("inf"), 0
    resume = cfg.train.get("resume_from")
    if resume:
        state = torch.load(str(resume), map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step, best = int(state["step"]), float(state["best"])
        first_epoch = int(state["epoch"]) + 1
    if distributed.primary:
        print({"world_size": distributed.world_size,
               "micro_batch_per_gpu": int(cfg.train.batch_size),
               "gradient_accumulation_steps": accumulation,
               "effective_batch_size": effective_batch,
               "optimizer_steps": total,
               "sequence_presentations": int(cfg.data.train_size) * int(cfg.train.epochs)},
              flush=True)
    use_bf16 = str(cfg.train.get("precision", "fp32")).lower() == "bf16"
    for epoch in range(first_epoch, cfg.train.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(train_loader):
            should_step = (micro_step + 1) % accumulation == 0
            sync = nullcontext() if should_step or distributed.world_size == 1 else model.no_sync()
            with sync:
                with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
                    loss = lm_loss(model, batch, device)
                    if cfg.train.get("rank_weight", 0) and "rank_tokens" in batch:
                        loss = loss + cfg.train.rank_weight * rank_loss(
                            model, batch, device
                        )
                (loss / accumulation).backward()
            if not should_step:
                continue
            for g in opt.param_groups:
                g["lr"] = cfg.train.lr * cosine_warmup(
                    step, total, cfg.train.warmup_steps
                )
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.train.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if distributed.primary and step % cfg.train.log_every == 0:
                logger.log(step, {"loss": loss.item()}, prefix="train/")
            step += 1
        barrier(distributed)
        if distributed.primary:
            raw_model.eval()
            with torch.no_grad():
                val_losses = [
                    lm_loss(raw_model, b, device).item()
                    for i, b in enumerate(val_loader)
                    if i < int(cfg.train.get("eval_batches", 40))
                ]
                vloss = sum(val_losses) / len(val_losses)
            logger.log(step, {"loss": vloss}, prefix="val/")
            previous = epoch * int(cfg.data.train_size)
            exposure = (epoch + 1) * int(cfg.data.train_size)
            improved = vloss < best
            best = min(best, vloss)
            ckpt = {
                "model": raw_model.state_dict(), "optimizer": opt.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "epoch": epoch, "step": step, "best": best,
                "examples_seen": exposure, "n_params": n_params,
            }
            torch.save(ckpt, out_dir / "last.pt")
            if improved:
                torch.save(ckpt, out_dir / "best.pt")
            for _ in crossed_milestones(previous, exposure, milestones):
                save_checkpoint(ckpt, out_dir, exposure)
            print(f"[epoch {epoch}] examples={exposure} val_loss={vloss:.4f}", flush=True)
        barrier(distributed)
    if logger is not None:
        logger.close()
    close(distributed)


if __name__ == "__main__":
    main()
