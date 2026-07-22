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

from textjepa.data.faithful_token_edits import (
    MASK_TOKEN, STEP_BOUNDARY_TOKEN, faithful_replacement_vocab,
)
from textjepa.models.masked_diffusion_lm import (
    MaskedDiffusionLM,
    select_terminal_buffers,
)
from textjepa.utils.checkpoint import build_dataset, collate_for


def make_cfg(args):
    data = OmegaConf.load(Path(__file__).parents[1] / "configs/data/igsm_real_token_edit.yaml")
    data.corruption_mode = "iterative_refinement"
    data.refinement_probability = 0.0
    data.trajectory_variants = 1
    data.eval_trajectory_variants = 1
    data.sample_transition = True
    data.content_only_actions = True
    data.replacement_only_vocab = True
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
    target = (
        batch["goal_buffer_tokens"][:, 0]
        if "goal_buffer_tokens" in batch else
        select_terminal_buffers(batch["buffer_tokens"], batch["step_mask"])
    ).to(device)
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
    parser.add_argument("--train-size", type=int, default=51_200_000)
    parser.add_argument("--val-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="effective number of independent problems")
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-floor", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=912)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--max-sequence-len", type=int, default=768)
    parser.add_argument("--eval-batches", type=int, default=16)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    if args.batch_size % args.microbatch_size:
        raise ValueError("microbatch-size must divide effective batch-size")
    accumulation = args.batch_size // args.microbatch_size
    vocab = faithful_replacement_vocab()
    cfg = make_cfg(args)
    train_loader = loader(cfg, vocab, "train", args.microbatch_size, False)
    val_loader = loader(cfg, vocab, "val", args.microbatch_size, False)
    model = MaskedDiffusionLM(
        len(vocab), vocab.pad_id, vocab.token_to_id[MASK_TOKEN],
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
        max_sequence_len=args.max_sequence_len,
        boundary_id=vocab.token_to_id[STEP_BOUNDARY_TOKEN],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    total_steps = args.max_steps
    best = math.inf
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            step = batch_index // accumulation
            if step >= total_steps:
                break
            clean, valid, response = clean_batch(model, batch, device)
            loss, _ = model.mdlm_loss(clean, valid, response)
            (loss / accumulation).backward()
            running += loss.item()
            if (batch_index + 1) % accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            warm = min(1.0, (step + 1) / max(args.warmup_steps, 1))
            cosine = 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warm * (
                    args.lr_floor + (1 - args.lr_floor) * cosine
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(model, val_loader, device, args.eval_batches)
        metrics.update(
            epoch=epoch,
            train_elbo=running / max(min(len(train_loader), total_steps * accumulation), 1),
            optimizer_steps=min(total_steps, len(train_loader) // accumulation),
            independent_batch_size=args.batch_size,
        )
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        payload = {
            "model": model.state_dict(), "args": vars(args),
            "vocab_size": len(vocab), "pad_id": vocab.pad_id,
            "mask_id": vocab.token_to_id[MASK_TOKEN],
            "information_regime": "prompt conditioned; response length and official step-boundary scaffold shared with sentence JEPA; clean response used only as training target",
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
