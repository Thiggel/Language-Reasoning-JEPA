#!/usr/bin/env python3
"""Decision-gate diagnostics for the two-level edit JEPA."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from textjepa.training.trainer import to_device
from textjepa.utils.checkpoint import build_dataset, collate_for, load_run


def normalized_error(pred, target):
    return (F.layer_norm(pred, pred.shape[-1:])
            - F.layer_norm(target, target.shape[-1:])).abs().mean(-1)


def masked_mean(value, mask):
    return value[mask].mean().item() if mask.any() else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batches", type=int, default=16)
    args = parser.parse_args()
    model, vocab, cfg = load_run(args.ckpt, args.device)
    dataset = build_dataset(cfg, vocab, "val", size=args.examples)
    loader = DataLoader(
        dataset, batch_size=min(int(cfg.train.batch_size), 8), shuffle=False,
        collate_fn=partial(collate_for(cfg), pad_id=vocab.pad_id),
    )
    values: dict[str, list[float]] = {}

    def add(name, value):
        if value == value:
            values.setdefault(name, []).append(float(value))

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.batches:
                break
            batch = to_device(batch, torch.device(args.device))
            out = model(batch)
            shuffled = dict(batch)
            for key in ("op", "edit_position", "edit_content_token"):
                shuffled[key] = batch[key].roll(1, dims=1)
            shuffled_out = model(shuffled)
            valid_step = out.step_mask

            token_pred = out.extras.get("token_predictions")
            if token_pred is not None:
                token_valid = (
                    out.extras["token_prediction_mask"]
                    & out.extras["token_target_mask"]
                    & valid_step.unsqueeze(-1)
                )
                add("token_matched_l1", masked_mean(normalized_error(
                    token_pred, out.extras["token_targets"]
                ), token_valid))
                shuffled_valid = (
                    shuffled_out.extras["token_prediction_mask"]
                    & shuffled_out.extras["token_target_mask"]
                    & valid_step.unsqueeze(-1)
                )
                add("token_shuffled_l1", masked_mean(normalized_error(
                    shuffled_out.extras["token_predictions"],
                    shuffled_out.extras["token_targets"],
                ), shuffled_valid))
                add("token_action_effect", masked_mean(
                    normalized_error(token_pred,
                                     shuffled_out.extras["token_predictions"]),
                    out.extras["token_prediction_mask"]
                    & shuffled_out.extras["token_prediction_mask"]
                    & valid_step.unsqueeze(-1),
                ))

            sentence_pred = out.extras.get("sentence_predictions")
            if sentence_pred is not None:
                target = out.extras["sentence_targets"]
                affected = out.extras["affected_sentence"]
                index = torch.arange(target.shape[-2], device=target.device)
                valid = out.extras["sentence_target_mask"] & valid_step.unsqueeze(-1)
                changed = valid & index.view(1, 1, -1).eq(affected.unsqueeze(-1))
                unchanged = valid & ~changed
                error = normalized_error(sentence_pred, target)
                shuffled_error = normalized_error(
                    shuffled_out.extras["sentence_predictions"], target
                )
                add("sentence_changed_matched_l1", masked_mean(error, changed))
                add("sentence_changed_shuffled_l1", masked_mean(
                    shuffled_error, changed
                ))
                add("sentence_unchanged_matched_l1", masked_mean(error, unchanged))
                add("sentence_action_effect", masked_mean(normalized_error(
                    sentence_pred,
                    shuffled_out.extras["sentence_predictions"]
                ), changed))
                teacher = out.extras["sentence_states_tgt"]
                delta = (teacher[:, 1:] - teacher[:, :-1]).pow(2).mean(-1).sqrt()
                add("target_changed_sentence_delta", masked_mean(delta, changed))
                add("target_unchanged_sentence_delta", masked_mean(delta, unchanged))
                copy_error = normalized_error(
                    out.extras["sentence_states"][:, :-1], target
                )
                add("sentence_changed_copy_l1", masked_mean(copy_error, changed))

            if out.hi_preds is not None:
                add("macro_matched_l1", masked_mean(
                    normalized_error(out.hi_preds, out.hi_targets), out.hi_mask
                ))
                add("macro_shuffled_l1", masked_mean(
                    normalized_error(shuffled_out.hi_preds, out.hi_targets),
                    out.hi_mask & shuffled_out.hi_mask,
                ))
            attention = out.extras["sentence_attention"]
            positive = attention > 0
            entropy = -(attention.clamp_min(1e-12).log() * attention).sum(-1)
            sentence_count = out.extras["sentence_states"].abs().sum(-1).gt(0).sum(-1)
            add("attention_entropy_per_sentence", (
                entropy / sentence_count.clamp_min(1)
            )[sentence_count > 0].mean().item())

    metrics = {name: sum(group) / len(group) for name, group in values.items()}
    for prefix in ("token", "sentence_changed", "macro"):
        matched = metrics.get(f"{prefix}_matched_l1")
        shuffled = metrics.get(f"{prefix}_shuffled_l1")
        if matched is not None and shuffled is not None:
            metrics[f"{prefix}_shuffled_over_matched"] = shuffled / max(matched, 1e-12)
    metrics.update({
        "variant": model.variant,
        "examples": min(args.examples, len(dataset)),
        "trainable_parameters": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
        "information_regime": "observed_action; no clean goal used by predictor",
        "shuffle_control": "cyclic time-shuffle of complete primitive actions",
    })
    Path(args.out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
