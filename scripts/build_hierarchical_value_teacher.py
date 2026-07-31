#!/usr/bin/env python3
"""Build listwise value replay from grounded oracle-search trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.evaluate_hierarchical_language_oracles import load_checkpoint
from textjepa.data.language_planning import (
    MODEL_REVISION,
    TRANSFORMERS_VERSION,
)
from textjepa.data.provenance import sha256_file
from textjepa.models.hierarchical_language_jepa import (
    HIERARCHICAL_LANGUAGE_ARCHITECTURE,
)
from textjepa.planning.hierarchical_language import (
    construct_value_teacher_from_rollouts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oracle-rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-temperature", type=float, default=0.1)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--step-cost", type=float, default=0.01)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument(
        "--metric",
        choices=("euclidean", "mahalanobis"),
        default="mahalanobis",
    )
    parser.add_argument(
        "--max-prefix", type=int, default=8,
        help="Total plan prefixes retained; 1 is the raw endpoint teacher.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if min(
        args.terminal_temperature, args.teacher_temperature
    ) <= 0 or min(args.step_cost, args.prior_weight) < 0 or (
        args.max_prefix < 1
    ):
        raise ValueError("invalid value-teacher cost configuration")
    model, learner = load_checkpoint(args.checkpoint, args.device)
    payload = torch.load(
        args.oracle_rollouts, map_location=args.device, weights_only=True
    )
    required = {
        "rollout_states", "rollout_log_probabilities", "rollout_mask",
        "goals", "goal_mask", "successor_state", "successor_context",
        "task_hidden", "first_action_log_probability", "action_mask",
        "dataset_fingerprint", "terminal_set_fingerprint",
        "symbolically_verified",
    }
    if not required <= payload.keys():
        raise ValueError(
            f"oracle rollout artifact lacks {sorted(required - payload.keys())}"
        )
    if payload["symbolically_verified"] is not True:
        raise ValueError("value teacher requires verified terminal sets")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    available = payload["rollout_states"].shape[-2]
    if args.max_prefix > available:
        raise ValueError("requested value prefix exceeds oracle rollouts")
    states = payload["rollout_states"][..., :args.max_prefix, :]
    log_probability = payload["rollout_log_probabilities"][
        ..., :args.max_prefix
    ]
    rollout_mask = payload["rollout_mask"][..., :args.max_prefix]
    metric = (
        learner.sentence_metric
        if args.metric == "mahalanobis"
        else lambda left, right: (left - right).square().sum(-1)
    )
    target = construct_value_teacher_from_rollouts(
        states,
        log_probability,
        rollout_mask,
        payload["goals"], payload["goal_mask"],
        metric,
        terminal_temperature=args.terminal_temperature,
        teacher_temperature=args.teacher_temperature,
        step_cost=args.step_cost,
        prior_weight=args.prior_weight,
    )
    output = {
        "successor_state": payload["successor_state"].cpu(),
        "successor_context": payload["successor_context"].cpu(),
        "task_hidden": payload["task_hidden"].cpu(),
        "first_action_log_probability": payload[
            "first_action_log_probability"
        ].cpu(),
        "teacher_cost": target.cpu(),
        "action_mask": payload["action_mask"].cpu(),
        "step_cost": args.step_cost,
        "prior_weight": args.prior_weight,
        "teacher_temperature": args.teacher_temperature,
        "value_temperature": args.teacher_temperature,
        "metric_name": args.metric,
        "teacher_kind": (
            "raw_endpoint" if args.max_prefix == 1
            else "supported_search_quasimetric"
        ),
        "max_total_prefix": args.max_prefix,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_stage": learner.stage.name,
        "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "dataset_fingerprint": payload["dataset_fingerprint"],
        "terminal_set_fingerprint": payload["terminal_set_fingerprint"],
        "oracle_rollout_sha256": sha256_file(args.oracle_rollouts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)


if __name__ == "__main__":
    main()
