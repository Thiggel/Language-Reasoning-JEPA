"""Audit the synthetic denoising data used by faithful token-edit JEPA.

This is a data audit, not a model evaluation.  It makes the distinction
between official iGSM source text and the synthetic corruption/repair process
explicit, checks exact recovery, and reports how much of the requested edit
trajectory is irreducible under token Levenshtein distance.

Example:
    .venv/bin/python scripts/audit_faithful_token_edit_data.py --examples 256
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from textjepa.data.faithful_token_edits import (
    FaithfulTokenEditDataset,
    faithful_token_edit_vocab,
)


def percentile(values: list[int | float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    fraction = position - low
    return float(ordered[low] * (1.0 - fraction) + ordered[high] * fraction)


def summary(values: list[int | float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": float(min(values)) if values else None,
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": float(max(values)) if values else None,
        "mean": float(sum(values) / len(values)) if values else None,
    }


def token_edit_distance(left: list[int], right: list[int]) -> int:
    """Memory-bounded Levenshtein distance over token ids."""
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_token in enumerate(right, start=1):
        current = [row]
        for column, left_token in enumerate(left, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + int(left_token != right_token),
            ))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-edits", type=int, default=6)
    parser.add_argument("--max-edits", type=int, default=16)
    parser.add_argument("--max-op", type=int, default=21)
    parser.add_argument("--max-edge", type=int, default=28)
    parser.add_argument("--op-min", type=int, default=8)
    parser.add_argument("--op-max", type=int, default=21)
    parser.add_argument("--counterfactual-k", type=int, default=0)
    parser.add_argument(
        "--counterfactual-source",
        choices=("uniform_local", "mixed"),
        default="uniform_local",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab,
        size=args.examples,
        seed=args.seed,
        max_op=args.max_op,
        max_edge=args.max_edge,
        op_range=(args.op_min, args.op_max),
        min_edits=args.min_edits,
        max_edits=args.max_edits,
        counterfactual_k=args.counterfactual_k,
        counterfactual_source=args.counterfactual_source,
    )

    metrics: dict[str, list[int | float]] = {
        "official_solution_tokens": [],
        "official_solution_steps": [],
        "maximum_official_step_tokens": [],
        "initial_buffer_chunks": [],
        "maximum_initial_chunk_tokens": [],
        "trajectory_edits": [],
        "initial_token_edit_distance": [],
        "trajectory_to_minimum_distance_ratio": [],
        "maximum_rendered_action_tokens": [],
    }
    operation_counts: Counter[str] = Counter()
    collapsed_buffers = 0
    exact_recoveries = 0
    counterfactual_candidates = 0
    counterfactual_states = 0

    for index in range(len(dataset)):
        item = dataset[index]
        source = dataset.source[index]
        # The gold target is intentionally not returned by the edit dataset;
        # recover it here only inside this offline audit from the official
        # source trace.
        target_buffer = [list(sentence) for sentence in source["steps"]]
        target = [token for sentence in target_buffer for token in sentence]
        initial = [token for chunk in item["buffers"][0] for token in chunk]
        terminal = [token for chunk in item["buffers"][-1] for token in chunk]
        distance = token_edit_distance(initial, target)
        edits = len(item["actions"])

        metrics["official_solution_tokens"].append(len(target))
        metrics["official_solution_steps"].append(len(source["steps"]))
        metrics["maximum_official_step_tokens"].append(
            max((len(step) for step in source["steps"]), default=0)
        )
        metrics["initial_buffer_chunks"].append(len(item["buffers"][0]))
        metrics["maximum_initial_chunk_tokens"].append(
            max((len(chunk) for chunk in item["buffers"][0]), default=0)
        )
        metrics["trajectory_edits"].append(edits)
        metrics["initial_token_edit_distance"].append(distance)
        metrics["trajectory_to_minimum_distance_ratio"].append(
            edits / max(distance, 1)
        )
        metrics["maximum_rendered_action_tokens"].append(
            max((len(action) for action in item["actions"]), default=0)
        )
        operation_counts.update(
            {0: "delete", 1: "insert", 2: "replace"}.get(op, str(op))
            for op in item["op"]
        )
        collapsed_buffers += int(
            len(source["steps"]) > 1 and len(item["buffers"][0]) == 1
        )
        exact_recoveries += int(terminal == target)

        alternatives = item.get("alt_actions", [])
        if alternatives:
            counterfactual_states += sum(bool(candidates) for candidates in alternatives)
            counterfactual_candidates += sum(len(candidates) for candidates in alternatives)

    total_ops = sum(operation_counts.values())
    payload = {
        "schema_version": 1,
        "examples": len(dataset),
        "source_contract": {
            "problem_and_gold_solution": "official iGSM",
            "buffer_corruption": "synthetic literal token edits",
            "repair_supervision": "oracle inverse-corruption stack",
            "corruption_token_pool": "tokens from the example's gold solution",
            "counterfactual_source": args.counterfactual_source,
            "same_state_counterfactuals_present": bool(counterfactual_candidates),
            "terminal_gold_tokens_exposed_to_model_batch": False,
            "terminal_buffer_used_as_training_target": True,
        },
        "exact_terminal_recovery_rate": exact_recoveries / max(len(dataset), 1),
        "multi_step_solutions_collapsed_to_one_chunk_rate": (
            collapsed_buffers / max(len(dataset), 1)
        ),
        "operation_fraction": {
            name: count / max(total_ops, 1)
            for name, count in sorted(operation_counts.items())
        },
        "counterfactual_candidates": counterfactual_candidates,
        "counterfactual_visited_states": counterfactual_states,
        "distributions": {name: summary(values) for name, values in metrics.items()},
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
