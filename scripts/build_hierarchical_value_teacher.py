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
from textjepa.data.provenance import artifact_fingerprint, sha256_file
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
    parser.add_argument("--ranking-entropy-fraction", type=float, default=0.60)
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
    ) <= 0 or not 0 < args.ranking_entropy_fraction < 1 or min(
        args.step_cost, args.prior_weight
    ) < 0 or (
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
        "terminal_set_symbolically_verified", "rollouts_exactly_grounded",
        "checkpoint_sha256", "oracle_rollout_fingerprint",
    }
    if not required <= payload.keys():
        raise ValueError(
            f"oracle rollout artifact lacks {sorted(required - payload.keys())}"
        )
    if payload["terminal_set_symbolically_verified"] is not True:
        raise ValueError("value teacher requires verified terminal sets")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if payload["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(
            "oracle rollouts were produced by a different macro checkpoint"
        )
    if payload["rollouts_exactly_grounded"] is not True:
        raise ValueError(
            "value distillation requires worker-executed, exactly re-encoded "
            "rollouts; latent prior rollouts are diagnostic-only"
        )
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
    # Interpret temperatures relative to each geometry's empirical terminal
    # scale. This prevents Euclidean and Mahalanobis cells from receiving
    # different target entropy merely because their coordinate scales differ.
    left = states[..., 0, :][..., None, :]
    right = payload["goals"][:, None, None, :, :]
    distance = metric(left, right).masked_fill(
        ~payload["goal_mask"][:, None, None, :], torch.inf
    ).amin(-1)
    positive = distance[torch.isfinite(distance) & distance.gt(0)]
    geometry_scale = float(positive.median()) if len(positive) else 1.0
    geometry_scale = max(geometry_scale, 1e-8)
    terminal_temperature = args.terminal_temperature * geometry_scale
    continuation_temperature = args.teacher_temperature * geometry_scale
    target = construct_value_teacher_from_rollouts(
        states,
        log_probability,
        rollout_mask,
        payload["goals"], payload["goal_mask"],
        metric,
        terminal_temperature=terminal_temperature,
        teacher_temperature=continuation_temperature,
        step_cost=args.step_cost,
        prior_weight=args.prior_weight,
    )
    action_mask = payload["action_mask"]
    action_counts = action_mask.sum(-1).clamp_min(1)
    desired_entropy = (
        action_counts.float().log() * args.ranking_entropy_fraction
    ).mean()
    lo, hi = geometry_scale * 1e-5, geometry_scale * 1e3
    for _ in range(60):
        ranking_temperature = (lo * hi) ** 0.5
        logits = (-target / ranking_temperature).masked_fill(
            ~action_mask, -torch.inf
        )
        probabilities = logits.softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean()
        if entropy < desired_entropy:
            lo = ranking_temperature
        else:
            hi = ranking_temperature
    ranking_temperature = (lo * hi) ** 0.5
    sorted_cost = target.masked_fill(~action_mask, torch.inf).sort(-1).values
    top_one_margin = (sorted_cost[:, 1] - sorted_cost[:, 0]).mean()
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
        "teacher_temperature": ranking_temperature,
        "value_temperature": ranking_temperature,
        "terminal_temperature": terminal_temperature,
        "continuation_softmin_temperature": continuation_temperature,
        "geometry_scale": geometry_scale,
        "target_entropy": float(entropy),
        "target_entropy_fraction": args.ranking_entropy_fraction,
        "mean_top_one_margin": float(top_one_margin),
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
        "oracle_rollout_fingerprint": payload["oracle_rollout_fingerprint"],
    }
    output["value_replay_fingerprint"] = artifact_fingerprint(output, (
        "successor_state", "successor_context", "task_hidden",
        "first_action_log_probability", "teacher_cost", "action_mask",
        "dataset_fingerprint", "terminal_set_fingerprint",
        "oracle_rollout_fingerprint", "source_checkpoint_sha256",
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)


if __name__ == "__main__":
    main()
