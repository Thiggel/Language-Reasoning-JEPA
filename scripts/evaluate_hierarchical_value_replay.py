#!/usr/bin/env python3
"""Evaluate listwise value ranking on a held-out grounded replay artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.data.provenance import artifact_fingerprint, sha256_file
from textjepa.training.hierarchical_language import ResearchStage
from textjepa.utils.language_planning_runtime import load_hierarchical_checkpoint


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(-1)
    result = torch.empty_like(values)
    ranks = torch.arange(values.shape[-1], device=values.device, dtype=values.dtype)
    result.scatter_(-1, order, ranks.expand_as(values))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--value-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model, learner = load_hierarchical_checkpoint(args.checkpoint, args.device)
    if learner.stage != ResearchStage.VALUE_DISTILLATION or model.v is None:
        raise ValueError("value evaluation requires VALUE_DISTILLATION")
    replay = torch.load(args.value_replay, map_location=args.device, weights_only=True)
    fields = (
        "successor_state", "successor_context", "task_hidden",
        "first_action_log_probability", "teacher_cost", "action_mask",
        "dataset_fingerprint", "terminal_set_fingerprint",
        "oracle_rollout_fingerprint", "source_checkpoint_sha256",
    )
    if replay.get("value_replay_fingerprint") != artifact_fingerprint(replay, fields):
        raise ValueError("held-out value replay fingerprint is invalid")
    mask = replay["action_mask"]
    task = model.task_projection(replay["task_hidden"].to(
        dtype=next(model.parameters()).dtype
    ))
    predicted = (
        float(replay["step_cost"])
        - float(replay["prior_weight"]) * replay["first_action_log_probability"]
        + model.v(replay["successor_state"], replay["successor_context"], task)
    )
    teacher = replay["teacher_cost"]
    teacher_masked = teacher.masked_fill(~mask, torch.inf)
    predicted_masked = predicted.masked_fill(~mask, torch.inf)
    selected = predicted_masked.argmin(-1)
    best = teacher_masked.amin(-1)
    regret = teacher.gather(-1, selected[:, None]).squeeze(-1) - best
    pairs = mask[:, :, None] & mask[:, None, :]
    teacher_delta = teacher[:, :, None] - teacher[:, None, :]
    predicted_delta = predicted[:, :, None] - predicted[:, None, :]
    decisive = pairs & teacher_delta.ne(0)
    pairwise = (
        (teacher_delta.sign() == predicted_delta.sign()) & decisive
    ).sum() / decisive.sum().clamp_min(1)
    teacher_rank = _rank(teacher_masked)
    predicted_rank = _rank(predicted_masked)
    centered_t = teacher_rank - teacher_rank.mean(-1, keepdim=True)
    centered_p = predicted_rank - predicted_rank.mean(-1, keepdim=True)
    spearman = (
        (centered_t * centered_p).sum(-1)
        / (centered_t.square().sum(-1) * centered_p.square().sum(-1)).sqrt().clamp_min(1e-12)
    ).mean()
    output = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "value_replay_fingerprint": replay["value_replay_fingerprint"],
        "dataset_fingerprint": replay["dataset_fingerprint"],
        "metrics": {
            "top_one_regret": float(regret.mean()),
            "top_one_accuracy": float(regret.eq(0).float().mean()),
            "pairwise_ranking_accuracy": float(pairwise),
            "spearman": float(spearman),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
