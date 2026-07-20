#!/usr/bin/env python3
"""Fixed-noise and iterative-recovery audit for reference MDLM."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_edit_mdlm import make_cfg
from textjepa.data.faithful_token_edits import faithful_token_edit_vocab
from textjepa.models.masked_diffusion_lm import MaskedDiffusionLM
from textjepa.utils.checkpoint import build_dataset, collate_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batches", type=int, default=16)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = argparse.Namespace(**payload["args"])
    saved.val_size = args.examples
    cfg = make_cfg(saved)
    vocab = faithful_token_edit_vocab()
    model = MaskedDiffusionLM(
        payload["vocab_size"], payload["pad_id"], payload["mask_id"],
        d_model=saved.d_model, n_layers=saved.layers, n_heads=saved.heads,
    )
    model.load_state_dict(payload["model"])
    model.to(args.device).eval()
    dataset = build_dataset(cfg, vocab, "val", size=args.examples)
    loader = DataLoader(
        dataset, batch_size=saved.batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(collate_for(cfg), pad_id=vocab.pad_id),
    )
    sums, count = {}, 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.batches:
                break
            prompt = batch["prompt_tokens"].to(args.device)
            target_buffer = batch["buffer_tokens"][:, -1].to(args.device)
            initial_buffer = batch["buffer_tokens"][:, 0].to(args.device)
            clean, valid, response = model.pack_clean(prompt, target_buffer)
            for fraction in (0.25, 0.5, 0.75, 1.0):
                generator = torch.Generator(device=args.device)
                generator.manual_seed(20_000 + batch_index)
                draw = torch.rand(clean.shape, generator=generator,
                                  device=args.device)
                noised, masked = model.corrupt(
                    clean, response, torch.full((len(clean),), fraction,
                                                device=args.device),
                    random=draw,
                )
                logits, _ = model.logits(noised, valid, response)
                pred = logits.argmax(-1)
                key = f"noise_{fraction:g}_masked_token_accuracy"
                sums[key] = sums.get(key, 0.0) + pred[masked].eq(
                    clean[masked]
                ).float().mean().item()
            for steps in (4, 8, 16):
                torch.manual_seed(30_000 + batch_index)
                sampled, _, sample_response = model.sample(
                    prompt, initial_buffer, steps=steps
                )
                correct = sampled.eq(clean)
                token = correct[sample_response].float().mean().item()
                exact = (correct | ~sample_response).all(-1).float().mean().item()
                sums[f"subs_{steps}_token_accuracy"] = (
                    sums.get(f"subs_{steps}_token_accuracy", 0.0) + token
                )
                sums[f"subs_{steps}_sequence_exact"] = (
                    sums.get(f"subs_{steps}_sequence_exact", 0.0) + exact
                )
            count += 1
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    metrics.update({
        "method": "reference_mdlm_absorbing_subs",
        "examples": min(args.examples, len(dataset)),
        "network_evaluation_depths": [4, 8, 16],
        "information_regime": payload["information_regime"],
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    })
    Path(args.out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
