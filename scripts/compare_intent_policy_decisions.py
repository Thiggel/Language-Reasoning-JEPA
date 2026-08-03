"""Compare aligned JEPA and LM decision errors on the same iGSM states."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from textjepa.analysis.intent_decisions import aggregate_rank_metrics, rank_metrics


def _load(specification: str):
    label, remainder = specification.split("=", 1)
    path_text, score = remainder.rsplit(",", 1)
    payload = json.loads(Path(path_text).read_text())
    rows = payload["rows"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["episode"], row["trace"], row["step"])].append(row)
    return label, score, grouped, payload


def _decision(candidates: list[dict], score: str) -> dict:
    ordered = sorted(candidates, key=lambda row: (row[score], row["action"]))
    metrics = rank_metrics(
        [row[score] for row in candidates],
        [row["necessary"] for row in candidates],
    )
    return {
        **metrics,
        "selected_action": ordered[0]["action"],
        "selected_operation": ordered[0]["operation"],
        "remaining_necessary": ordered[0]["remaining_necessary"],
        "step": ordered[0]["step"],
        "trace": ordered[0]["trace"],
    }


def _bucket(values, key, buckets):
    result = {}
    for lower, upper in buckets:
        selected = [row for row in values if lower <= row[key] <= upper]
        if selected:
            result[f"{lower}-{upper}"] = {
                "n": len(selected),
                "top1": float(np.mean([row["top1"] for row in selected])),
                "pairwise_accuracy": float(np.mean([
                    row["pairwise_accuracy"] for row in selected
                    if row["n_candidates"] > row["n_positive"]
                ])) if any(
                    row["n_candidates"] > row["n_positive"] for row in selected
                ) else None,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", action="append", required=True,
        help="LABEL=AUDIT_JSON,SCORE_FIELD",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    models = {}
    candidate_sets = {}
    payloads = {}
    for specification in args.model:
        label, score, groups, payload = _load(specification)
        models[label] = {
            key: _decision(candidates, score) for key, candidates in groups.items()
        }
        candidate_sets[label] = {
            key: tuple(sorted(row["action"] for row in candidates))
            for key, candidates in groups.items()
        }
        payloads[label] = payload
    state_sets = {label: set(value) for label, value in models.items()}
    reference_label = next(iter(state_sets))
    reference_states = state_sets[reference_label]
    for label, states in state_sets.items():
        if states != reference_states:
            raise RuntimeError(
                f"unaligned decision states: {reference_label} has "
                f"{len(reference_states)}, {label} has {len(states)}"
            )
    common = reference_states
    if not common:
        raise RuntimeError("audits have no aligned decision states")
    for key in common:
        reference_candidates = candidate_sets[reference_label][key]
        for label in candidate_sets:
            if candidate_sets[label][key] != reference_candidates:
                raise RuntimeError(
                    f"unaligned feasible menu at state {key}: "
                    f"{reference_label} versus {label}"
                )
    result = {
        "aligned_states": len(common),
        "models": {},
        "error_overlap": {},
        "information_boundary": (
            "All audits must share dataset split, seed, prompt seed, and feasible "
            "action menu. Correctness labels are post-hoc only."
        ),
    }
    for label, decisions in models.items():
        selected = [decisions[key] for key in sorted(common)]
        by_trace = {}
        for trace in sorted({row["trace"] for row in selected}):
            rows = [row for row in selected if row["trace"] == trace]
            by_trace[trace] = {
                "summary": aggregate_rank_metrics(rows),
                "by_remaining": _bucket(
                    rows, "remaining_necessary", ((1, 2), (3, 4), (5, 9))
                ),
                "by_step": _bucket(rows, "step", ((0, 1), (2, 3), (4, 9))),
                "selected_operation": {
                    operation: {
                        "n": sum(
                            row["selected_operation"] == operation for row in rows
                        ),
                        "top1": float(np.mean([
                            row["top1"] for row in rows
                            if row["selected_operation"] == operation
                        ])),
                    }
                    for operation in sorted({
                        row["selected_operation"] for row in rows
                    })
                },
            }
        result["models"][label] = by_trace
    for left, left_decisions in models.items():
        result["error_overlap"][left] = {}
        left_errors = {key for key in common if not left_decisions[key]["top1"]}
        for right, right_decisions in models.items():
            right_errors = {
                key for key in common if not right_decisions[key]["top1"]
            }
            union = left_errors | right_errors
            result["error_overlap"][left][right] = {
                "both_wrong": len(left_errors & right_errors),
                "left_only": len(left_errors - right_errors),
                "right_only": len(right_errors - left_errors),
                "jaccard": (
                    len(left_errors & right_errors) / len(union) if union else 1.0
                ),
            }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
