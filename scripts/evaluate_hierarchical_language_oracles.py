#!/usr/bin/env python3
"""Evaluate the three distinct oracle goals from offline candidate tensors."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter

import torch

from textjepa.models.hierarchical_language_jepa import (
    HIERARCHICAL_LANGUAGE_ARCHITECTURE,
    HierarchicalLanguageJEPA,
    HierarchicalLanguageJEPAConfig,
)
from textjepa.data.provenance import artifact_fingerprint, sha256_file
from textjepa.planning.hierarchical_language import (
    exact_endpoint_control,
    oracle_high_level_prefix_cost,
    score_token_space_candidates,
    value_guided_high_level_prefix_cost,
)
from textjepa.training.hierarchical_language import (
    HierarchicalLanguageLearner,
    ResearchStage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["flat_token", "sentence_worker", "oracle_terminal", "value"],
        required=True,
    )
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--metric",
        choices=("mahalanobis", "euclidean", "cosine", "normalized_euclidean"),
        default="mahalanobis",
    )
    parser.add_argument("--step-cost", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-depth-matrix", action="store_true")
    parser.add_argument("--allow-unbound-candidates", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != HIERARCHICAL_LANGUAGE_ARCHITECTURE:
        raise ValueError(
            "checkpoint does not use the strict nested E0 -> E0_to_1 tower"
        )
    config = HierarchicalLanguageJEPAConfig(**checkpoint["config"])
    model = HierarchicalLanguageJEPA(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    stage = ResearchStage[checkpoint["stage"]]
    learner = HierarchicalLanguageLearner(model, stage).to(device).eval()
    learner.load_state_dict(checkpoint["learner"])
    return model, learner


def _metric(name: str, mahalanobis):
    if name == "mahalanobis":
        return mahalanobis
    if name == "euclidean":
        return lambda left, right: (left - right).square().sum(-1)
    if name == "cosine":
        return lambda left, right: 1 - torch.nn.functional.cosine_similarity(
            left, right, dim=-1
        )
    return lambda left, right: (
        torch.nn.functional.normalize(left, dim=-1)
        - torch.nn.functional.normalize(right, dim=-1)
    ).square().sum(-1)


def validate_planning_depth_matrix(
    rows: list[dict], mode: str, expected_grid: dict | None
) -> None:
    if not isinstance(expected_grid, dict):
        raise ValueError(
            "complete planning-depth validation requires expected_grid"
        )
    common = {"dataset_split", "symbolic_depth", "population_size"}
    mode_dimensions = {
        "flat_token": {
            "token_oracle_horizon", "endpoint_kind",
        },
        "sentence_worker": {"k0", "endpoint_kind"},
        "oracle_terminal": {"k1", "cem_iterations"},
        "value": {"k1", "cem_iterations"},
    }[mode]
    dimensions = sorted(common | mode_dimensions)
    if set(expected_grid) != set(dimensions):
        raise ValueError(
            f"expected_grid dimensions must be exactly {dimensions}"
        )
    canonical = {
        ("flat_token", "token_oracle_horizon"): {4, 8, 16},
        ("sentence_worker", "k0"): {8, 16, 32, 64},
        ("oracle_terminal", "k1"): {1, 2, 4, 8},
        ("value", "k1"): {1, 2, 4, 8},
    }
    for key, required in canonical.items():
        if key[0] == mode and set(expected_grid[key[1]]) != required:
            raise ValueError(f"{key[1]} does not match the canonical grid")
    if mode in {"flat_token", "sentence_worker"} and set(
        expected_grid["endpoint_kind"]
    ) != {"paired"}:
        raise ValueError("endpoint_kind must record paired exact/predicted scores")
    expected = set(itertools.product(*[
        expected_grid[name] for name in dimensions
    ]))
    observed = {
        tuple(row[name] for name in dimensions) for row in rows
    }
    if observed != expected:
        raise ValueError(
            "planning-depth matrix is incomplete or contains extra cells: "
            f"missing={len(expected - observed)}, "
            f"extra={len(observed - expected)}"
        )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.prior_weight < 0 or args.temperature <= 0 or args.step_cost < 0:
        raise ValueError("planning costs and temperature are invalid")
    model, learner = load_checkpoint(args.checkpoint, args.device)
    required_stage = {
        "flat_token": ResearchStage.TOKEN_JEPA,
        "sentence_worker": ResearchStage.SENTENCE_JEPA,
        "oracle_terminal": ResearchStage.MACRO_ACTION,
        "value": ResearchStage.VALUE_DISTILLATION,
    }[args.mode]
    if learner.stage < required_stage:
        raise ValueError(
            f"{args.mode} requires at least {required_stage.name}, got "
            f"{learner.stage.name}"
        )
    raw_batch = torch.load(
        args.candidates, map_location="cpu", weights_only=True
    )
    provenance_fields = {
        "flat_token": (
            "candidate_tokens", "candidate_log_probability",
            "predicted_endpoints", "exact_endpoints", "waypoint",
            "reference_tokens", "metadata",
        ),
        "sentence_worker": (
            "candidate_tokens", "candidate_log_probability",
            "predicted_endpoints", "exact_endpoints", "waypoint",
            "candidate_boundary_complete",
            "candidate_symbolic_state_id", "waypoint_symbolic_state_id",
            "metadata",
        ),
        "oracle_terminal": (
            "states", "log_probabilities", "goals", "metadata",
        ),
        "value": (
            "states", "contexts", "task_embedding",
            "log_probabilities", "metadata",
        ),
    }[args.mode]
    if not args.allow_unbound_candidates:
        if (
            raw_batch.get("architecture")
            != HIERARCHICAL_LANGUAGE_ARCHITECTURE
            or raw_batch.get("checkpoint_sha256")
            != sha256_file(args.checkpoint)
            or not raw_batch.get("dataset_fingerprint")
        ):
            raise ValueError(
                "candidate artifact is not bound to checkpoint and dataset"
            )
        if raw_batch.get("candidate_payload_fingerprint") != (
            artifact_fingerprint(raw_batch, provenance_fields)
        ):
            raise ValueError("candidate artifact fingerprint is invalid")
    batch = {
        name: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for name, value in raw_batch.items()
    }
    rows = []
    if args.mode in {"flat_token", "sentence_worker"}:
        if args.mode == "sentence_worker":
            validity = batch.get("nested_validity")
            required_metrics = {
                "heldout_sentence_dynamics",
                "sentence_effective_rank",
                "symbolic_state_purity",
            }
            if not isinstance(validity, dict) or validity.get("passed") is not True:
                raise ValueError("sentence worker requires a passed nested-state gate")
            if not required_metrics <= set(validity.get("metrics", {})):
                raise ValueError("nested-state validity metrics are incomplete")
            if not args.allow_unbound_candidates and (
                validity.get("checkpoint_sha256")
                != raw_batch["checkpoint_sha256"]
                or validity.get("dataset_fingerprint")
                != raw_batch["dataset_fingerprint"]
                or validity.get("candidate_payload_fingerprint")
                != raw_batch["candidate_payload_fingerprint"]
                or validity.get("architecture")
                != HIERARCHICAL_LANGUAGE_ARCHITECTURE
            ):
                raise ValueError(
                    "nested-state validity is not bound to this evaluation"
                )
            for name in (
                "candidate_boundary_complete",
                "candidate_symbolic_state_id",
                "waypoint_symbolic_state_id",
            ):
                if name not in batch:
                    raise ValueError(f"sentence worker requires {name}")
        for root in range(len(batch["candidate_tokens"])):
            started = perf_counter()
            common = dict(
                candidate_tokens=batch["candidate_tokens"][root],
                candidate_log_probability=batch["candidate_log_probability"][root],
                metric=_metric(
                    args.metric,
                    learner.token_metric
                    if args.mode == "flat_token"
                    else learner.sentence_metric,
                ),
                prior_weight=args.prior_weight,
                temperature=args.temperature,
                vocab_size=model.config.vocab_size,
            )
            if args.mode == "flat_token":
                predicted = score_token_space_candidates(
                    predicted_endpoints=batch["predicted_endpoints"][root],
                    token_waypoint=batch["waypoint"][root],
                    **common,
                )
                exact = score_token_space_candidates(
                    predicted_endpoints=batch["exact_endpoints"][root],
                    token_waypoint=batch["waypoint"][root],
                    **common,
                )
            else:
                control = exact_endpoint_control(
                    predicted_endpoints=batch["predicted_endpoints"][root],
                    exact_endpoints=batch["exact_endpoints"][root],
                    waypoint=batch["waypoint"][root],
                    sentence_encoder=model.e0_to_1,
                    **common,
                )
                predicted, exact = control.predicted, control.exact
            rows.append({
                "root": root,
                "predicted_best": predicted.best_index,
                "exact_best": exact.best_index,
                "predicted_cost": float(predicted.costs.min()),
                "selected_exact_cost": float(
                    exact.costs[predicted.best_index]
                ),
                "best_exact_cost": float(exact.costs.min()),
                "planning_wall_seconds": perf_counter() - started,
            })
            lm_best = int(
                batch["candidate_log_probability"][root].argmax()
            )
            rows[-1]["lm_baseline_exact_cost"] = float(
                exact.costs[lm_best]
            )
            rows[-1]["exact_oracle_gain"] = float(
                exact.costs[lm_best] - exact.costs.min()
            )
            if args.mode == "flat_token" and "reference_tokens" in batch:
                reference = batch["reference_tokens"][root]
                candidates = batch["candidate_tokens"][root]
                rows[-1].update({
                    "greedy_span_accuracy": float(torch.equal(
                        candidates[0], reference
                    )),
                    "proposal_oracle_span_accuracy": float(
                        (candidates == reference[None]).all(-1).any()
                    ),
                    "greedy_next_token_accuracy": float(
                        candidates[0, 0] == reference[0]
                    ),
                    "proposal_oracle_next_token_accuracy": float(
                        (candidates[:, 0] == reference[0]).any()
                    ),
                    "predicted_span_accuracy": float(torch.equal(
                        candidates[predicted.best_index], reference
                    )),
                    "exact_span_accuracy": float(torch.equal(
                        candidates[exact.best_index], reference
                    )),
                    "predicted_next_token_accuracy": float(
                        candidates[predicted.best_index, 0]
                        == reference[0]
                    ),
                    "exact_next_token_accuracy": float(
                        candidates[exact.best_index, 0]
                        == reference[0]
                    ),
                })
            if args.mode == "sentence_worker":
                completed = batch["candidate_boundary_complete"][root]
                if completed.dtype != torch.bool or not bool(completed.all()):
                    raise ValueError(
                        "sentence-worker candidates must end at real boundaries"
                    )
                symbolic = batch["candidate_symbolic_state_id"][root]
                goal_symbolic = batch["waypoint_symbolic_state_id"][root]
                rows[-1].update({
                    "predicted_selected_symbolic_success": bool(
                        symbolic[predicted.best_index] == goal_symbolic
                    ),
                    "exact_selected_symbolic_success": bool(
                        symbolic[exact.best_index] == goal_symbolic
                    ),
                })
    elif args.mode == "oracle_terminal":
        cost, horizon = oracle_high_level_prefix_cost(
            batch["states"], batch["log_probabilities"],
            batch["goals"], learner.sentence_metric,
            terminal_temperature=args.temperature,
            step_cost=args.step_cost, prior_weight=args.prior_weight,
            goal_mask=batch.get("goal_mask"),
        )
        rows = [
            {"root": index, "cost": float(cost[index]),
             "selected_prefix_length": int(horizon[index]) + 1}
            for index in range(len(cost))
        ]
    else:
        if model.value is None:
            raise ValueError("value mode requires a checkpoint with a value head")
        cost, horizon = value_guided_high_level_prefix_cost(
            batch["states"], batch["contexts"], batch["task_embedding"],
            batch["log_probabilities"], model.value,
            step_cost=args.step_cost, prior_weight=args.prior_weight,
        )
        rows = [
            {"root": index, "cost": float(cost[index]),
             "selected_prefix_length": int(horizon[index]) + 1}
            for index in range(len(cost))
        ]
    metadata = batch.get("metadata")
    if args.require_depth_matrix and (
        not isinstance(metadata, list) or len(metadata) != len(rows)
    ):
        raise ValueError("planning-depth evaluation requires root metadata")
    if isinstance(metadata, list):
        for row, meta in zip(rows, metadata):
            row.update(meta)
    if args.require_depth_matrix:
        validate_planning_depth_matrix(
            rows, args.mode, batch.get("expected_grid")
        )
    metrics = {}
    if rows and args.mode == "flat_token":
        metrics = {
            "exact_oracle_gain": sum(
                row["exact_oracle_gain"] for row in rows
            ) / len(rows),
            "model_oracle_gap": sum(
                row["selected_exact_cost"] - row["best_exact_cost"]
                for row in rows
            ) / len(rows),
        }
        for name in (
            "greedy_span_accuracy", "proposal_oracle_span_accuracy",
            "greedy_next_token_accuracy",
            "proposal_oracle_next_token_accuracy",
            "predicted_span_accuracy", "exact_span_accuracy",
            "predicted_next_token_accuracy", "exact_next_token_accuracy",
        ):
            if name in rows[0]:
                metrics[name] = sum(row[name] for row in rows) / len(rows)
    elif rows and args.mode == "sentence_worker":
        metrics = {
            "exact_symbolic_waypoint_success": sum(
                row["exact_selected_symbolic_success"] for row in rows
            ) / len(rows),
            "predicted_symbolic_waypoint_success": sum(
                row["predicted_selected_symbolic_success"] for row in rows
            ) / len(rows),
            "latent_symbolic_disagreement": sum(
                row["predicted_selected_symbolic_success"]
                != row["exact_selected_symbolic_success"]
                for row in rows
            ) / len(rows),
            "model_oracle_gap": sum(
                row["selected_exact_cost"] - row["best_exact_cost"]
                for row in rows
            ) / len(rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({
            "mode": args.mode,
            "metric_name": args.metric,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "dataset_fingerprint": raw_batch.get("dataset_fingerprint"),
            "candidate_payload_fingerprint": raw_batch.get(
                "candidate_payload_fingerprint"
            ),
            "metrics": metrics,
            "rows": rows,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
