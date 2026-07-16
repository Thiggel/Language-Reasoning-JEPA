"""Train the sentence-latent LM baseline.

    python scripts/train_sentlm.py run_name=sent_lm
    python scripts/train_sentlm.py run_name=sent_lm_latent model.latent_target=true
"""

from __future__ import annotations

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
from textjepa.data.sampling import FreshEpochSampler
from textjepa.models.sent_lm import SentenceLM
from textjepa.training.loggers import MetricLogger
from textjepa.training.optim import build_optimizer, cosine_warmup
from textjepa.training.trainer import to_device
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, collate_for


@hydra.main(config_path="../configs", config_name="sentlm", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device)
    if cfg.data.get("name", "igsm") == "igsm_real":
        from textjepa.data.faithful import cached_faithful_vocab

        vocab = cached_faithful_vocab()
    else:
        vocab = build_vocab(cfg.data.modulus)
    target_kind = cfg.train.get("target_kind", "outcome")
    if target_kind == "intent":
        coll = partial(collate_intent_sentence_policy, pad_id=vocab.pad_id)
        wrap = IntentSentencePolicyDataset
    elif target_kind == "outcome":
        coll = partial(collate_for(cfg), pad_id=vocab.pad_id)
        wrap = lambda dataset: dataset
    else:
        raise ValueError(f"unknown sentence LM target_kind: {target_kind}")
    train_ds = wrap(build_dataset(cfg, vocab, split="train"))
    train_sampler = FreshEpochSampler(train_ds, seed=cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size,
        sampler=train_sampler, num_workers=cfg.train.num_workers, collate_fn=coll,
        drop_last=True, persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        wrap(build_dataset(cfg, vocab, split="val")),
        batch_size=cfg.train.batch_size,
        num_workers=2, collate_fn=coll,
    )
    model = SentenceLM(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SentenceLM parameters: {n_params / 1e6:.2f}M")

    opt = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    total = cfg.train.epochs * len(train_loader)
    logger = MetricLogger(out_dir)
    step, best = 0, float("inf")
    for epoch in range(cfg.train.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            for g in opt.param_groups:
                g["lr"] = cfg.train.lr * cosine_warmup(
                    step, total, cfg.train.warmup_steps
                )
            losses = model(to_device(batch, device))
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            if step % cfg.train.log_every == 0:
                logger.log(step, {k: v.item() for k, v in losses.items()},
                           prefix="train/")
            step += 1
        model.eval()
        with torch.no_grad():
            val_losses = [
                sum(model(to_device(b, device)).values()).item()
                for i, b in enumerate(val_loader) if i < 40
            ]
            vloss = sum(val_losses) / len(val_losses)
        logger.log(step, {"loss": vloss}, prefix="val/")
        ckpt = {
            "model": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch, "n_params": n_params,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if vloss < best:
            best = vloss
            torch.save(ckpt, out_dir / "best.pt")
        print(f"[epoch {epoch}] val_loss={vloss:.4f}", flush=True)
    logger.close()


if __name__ == "__main__":
    main()
