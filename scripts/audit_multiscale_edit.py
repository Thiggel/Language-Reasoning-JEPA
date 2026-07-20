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

            position_logits = out.extras.get("refinement_position_logits")
            if position_logits is not None:
                prior_valid = valid_step & batch["op"][:, :valid_step.shape[1]].eq(2)
                add("base_prior_position_accuracy", (
                    position_logits.argmax(-1)[prior_valid]
                    == batch["edit_position"][:, :valid_step.shape[1]][prior_valid]
                ).float().mean().item())
                content_logits = out.extras["refinement_content_logits"]
                add("base_prior_content_accuracy", (
                    content_logits.argmax(-1)[prior_valid]
                    == batch["edit_content_token"][:, :valid_step.shape[1]][prior_valid]
                ).float().mean().item())

            q = out.extras.get("base_action_value")
            q_target = out.extras.get("base_action_value_target")
            if q_target is not None:
                add("base_action_value_mae", (q[valid_step] - q_target[valid_step])
                    .abs().mean().item())
                add("base_action_value_sign_accuracy", (
                    q[valid_step].gt(0) == q_target[valid_step].gt(0)
                ).float().mean().item())
            alt_q = out.extras.get("base_alt_action_value")
            if alt_q is not None:
                alt_valid = out.extras["base_alt_action_valid"]
                alt_target = out.extras["base_alt_action_target"]
                add("base_alt_action_value_mae", (
                    alt_q[alt_valid] - alt_target[alt_valid]
                ).abs().mean().item())
                add("base_alt_action_value_sign_accuracy", (
                    alt_q[alt_valid].gt(0) == alt_target[alt_valid].gt(0)
                ).float().mean().item())
            state_value = out.extras.get("state_goal_distance_prediction")
            if state_value is not None:
                state_valid = out.extras["state_goal_distance_mask"]
                state_target = out.extras["state_goal_distance_target"]
                add("state_goal_distance_mae", (
                    state_value[state_valid] - state_target[state_valid]
                ).abs().mean().item())
            decoder_position = out.extras.get("macro_decoder_position_logits")
            if decoder_position is not None:
                decoder_valid = out.extras["macro_decoder_valid"]
                add("macro_decoder_position_accuracy", (
                    decoder_position.argmax(-1)[decoder_valid]
                    == out.extras["macro_decoder_position_target"][decoder_valid]
                ).float().mean().item())
                decoder_content = out.extras["macro_decoder_content_logits"]
                add("macro_decoder_content_accuracy", (
                    decoder_content.argmax(-1)[decoder_valid]
                    == out.extras["macro_decoder_content_target"][decoder_valid]
                ).float().mean().item())

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
