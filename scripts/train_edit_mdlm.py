#!/usr/bin/env python3
"""Train and audit a paper-faithful reference MDLM on sequence-edit iGSM."""

from __future__ import annotations

import argparse
import json
import math
import random
from functools import partial
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.faithful_token_edits import MASK_TOKEN, faithful_token_edit_vocab
from textjepa.models.masked_diffusion_lm import MaskedDiffusionLM
from textjepa.utils.checkpoint import build_dataset, collate_for


def make_cfg(args):
    data = OmegaConf.load(Path(__file__).parents[1] / "configs/data/igsm_real_token_edit.yaml")
    data.corruption_mode = "iterative_refinement"
    data.refinement_probability = 0.0
    data.trajectory_variants = 1
    data.eval_trajectory_variants = 1
    data.train_size = args.train_size
    data.val_size = args.val_size
    data.fresh_per_epoch = False
    return OmegaConf.create({"data": data})


def loader(cfg, vocab, split, batch_size, shuffle):
    dataset = build_dataset(cfg, vocab, split)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0,
        collate_fn=partial(collate_for(cfg), pad_id=vocab.pad_id),
        drop_last=shuffle,
    )


def clean_batch(model, batch, device):
    prompt = batch["prompt_tokens"].to(device)
    target = batch["buffer_tokens"][:, -1].to(device)
    return model.pack_clean(prompt, target)


@torch.no_grad()
def evaluate(model, data_loader, device, max_batches=16):
    model.eval()
    loss_sum = token_correct = token_total = 0.0
    seen = 0
    for index, batch in enumerate(data_loader):
        if index >= max_batches:
            break
        clean, valid, response = clean_batch(model, batch, device)
        # A fixed mid-time diagnostic makes checkpoints comparable; training
        # still samples continuous time uniformly.
        noise = torch.full((len(clean),), 0.5, device=device)
        torch.manual_seed(10_000 + index)
        loss, extra = model.mdlm_loss(clean, valid, response, noise)
        prediction = extra["logits"].argmax(-1)
        masked = extra["masked"]
        token_correct += prediction[masked].eq(clean[masked]).sum().item()
        token_total += masked.sum().item()
        loss_sum += loss.item()
        seen += 1
    return {
        "midtime_elbo": loss_sum / max(seen, 1),
        "midtime_masked_token_accuracy": token_correct / max(token_total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=16)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    vocab = faithful_token_edit_vocab()
    cfg = make_cfg(args)
    train_loader = loader(cfg, vocab, "train", args.batch_size, True)
    val_loader = loader(cfg, vocab, "val", args.batch_size, False)
    model = MaskedDiffusionLM(
        len(vocab), vocab.pad_id, vocab.token_to_id[MASK_TOKEN],
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    best = math.inf
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch_index, batch in enumerate(train_loader):
            clean, valid, response = clean_batch(model, batch, device)
            loss, _ = model.mdlm_loss(clean, valid, response)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            step = epoch * len(train_loader) + batch_index
            warm = min(1.0, (step + 1) / max(args.warmup_steps, 1))
            cosine = 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warm * cosine
            optimizer.step()
            running += loss.item()
        metrics = evaluate(model, val_loader, device, args.eval_batches)
        metrics.update(epoch=epoch, train_elbo=running / max(len(train_loader), 1))
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        payload = {
            "model": model.state_dict(), "args": vars(args),
            "vocab_size": len(vocab), "pad_id": vocab.pad_id,
            "mask_id": vocab.token_to_id[MASK_TOKEN],
            "information_regime": "prompt conditioned; response length/shape observed; clean response used only as training target",
        }
        torch.save(payload, out / "model/last.pt")
        if metrics["midtime_elbo"] < best:
            best = metrics["midtime_elbo"]
            torch.save(payload, out / "model/best.pt")
    (out / "training_metrics.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
