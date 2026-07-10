"""Train the decoder-only LM baseline (teacher-forced next-token CE).

    python scripts/train_lm.py +experiment=lm_9m
"""

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
from textjepa.models.lm_baseline import DecoderLM
from textjepa.training.loggers import MetricLogger
from textjepa.training.optim import build_optimizer, cosine_warmup
from textjepa.utils import seed_everything


def lm_loss(model, batch, device):
    tokens = batch["tokens"].to(device)
    logits = model(tokens)[:, :-1]
    tgt = tokens[:, 1:]
    pos = torch.arange(tgt.shape[1], device=device).unsqueeze(0)
    mask = (pos >= (batch["prompt_len"].to(device).unsqueeze(1) - 1)) & (
        tgt != model.pad_id
    )
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none"
    ).reshape(tgt.shape)
    return (ce * mask).sum() / mask.sum().clamp(min=1)


@hydra.main(config_path="../configs", config_name="lm", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device)
    vocab = build_vocab(cfg.data.modulus)

    def make(split):
        d = cfg.data
        size = d.val_size if split == "val" else d.train_size
        seed = d.val_seed if split == "val" else d.train_seed
        return LMDataset(
            vocab, size=size, seed=seed, modulus=d.modulus,
            n_vars_range=tuple(d.n_vars_range), leaf_prob=d.leaf_prob,
            steps_range=tuple(d.steps_range), distractor_prob=d.distractor_prob,
            max_distractors=d.max_distractors,
        )

    coll = partial(collate_lm, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        make("train"), batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, collate_fn=coll, drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        make("val"), batch_size=cfg.train.batch_size, num_workers=2,
        collate_fn=coll,
    )

    model = DecoderLM(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
        ff_mult=cfg.model.ff_mult, max_len=cfg.model.max_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LM parameters: {n_params / 1e6:.2f}M")

    opt = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    total = cfg.train.epochs * len(train_loader)
    logger = MetricLogger(out_dir)
    step, best = 0, float("inf")
    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in train_loader:
            for g in opt.param_groups:
                g["lr"] = cfg.train.lr * cosine_warmup(
                    step, total, cfg.train.warmup_steps
                )
            loss = lm_loss(model, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            if step % cfg.train.log_every == 0:
                logger.log(step, {"loss": loss.item()}, prefix="train/")
            step += 1
        model.eval()
        with torch.no_grad():
            vloss = sum(
                lm_loss(model, b, device).item()
                for i, b in enumerate(val_loader) if i < 40
            ) / 40
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
