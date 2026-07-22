"""Train the sentence-latent LM baseline.

    python scripts/train_sentlm.py run_name=sent_lm
    python scripts/train_sentlm.py run_name=sent_lm_latent model.latent_target=true
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import (
    IntentSentencePolicyDataset,
    collate_intent_sentence_policy,
)
from textjepa.data.sampling import DistributedFreshEpochSampler, FreshEpochSampler
from textjepa.models.sent_lm import SentenceLM
from textjepa.training.loggers import MetricLogger
from textjepa.training.optim import build_optimizer, cosine_warmup
from textjepa.training.distributed import barrier, close, initialize, wrap as ddp_wrap
from textjepa.training.scale import (
    crossed_milestones, exposure_milestones, optimizer_steps, save_checkpoint,
)
from textjepa.training.trainer import to_device
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, collate_for


@hydra.main(config_path="../configs", config_name="sentlm", version_base="1.3")
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
    target_kind = cfg.train.get("target_kind", "outcome")
    if target_kind == "intent":
        coll = partial(collate_intent_sentence_policy, pad_id=vocab.pad_id)
        dataset_wrap = IntentSentencePolicyDataset
    elif target_kind == "outcome":
        coll = partial(collate_for(cfg), pad_id=vocab.pad_id)
        dataset_wrap = lambda dataset: dataset
    else:
        raise ValueError(f"unknown sentence LM target_kind: {target_kind}")
    train_ds = dataset_wrap(build_dataset(cfg, vocab, split="train"))
    train_sampler = (
        DistributedFreshEpochSampler(
            train_ds, distributed.rank, distributed.world_size, seed=cfg.seed
        ) if distributed.world_size > 1 else FreshEpochSampler(train_ds, seed=cfg.seed)
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size,
        sampler=train_sampler, num_workers=cfg.train.num_workers, collate_fn=coll,
        drop_last=True, persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        dataset_wrap(build_dataset(cfg, vocab, split="val")),
        batch_size=cfg.train.batch_size,
        num_workers=int(cfg.train.get("val_num_workers", cfg.train.num_workers)),
        collate_fn=coll,
    )
    raw_model = SentenceLM(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    if distributed.primary:
        print(f"SentenceLM parameters: {n_params / 1e6:.2f}M")
    model = ddp_wrap(raw_model, distributed)

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
                    losses = model(to_device(batch, device))
                    loss = sum(losses.values())
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
                logger.log(step, {k: v.item() for k, v in losses.items()},
                           prefix="train/")
            step += 1
        barrier(distributed)
        if distributed.primary:
            raw_model.eval()
            with torch.no_grad():
                val_losses = [
                    sum(raw_model(to_device(b, device)).values()).item()
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
