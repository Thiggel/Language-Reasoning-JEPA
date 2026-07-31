#!/usr/bin/env python3
"""Plot task accuracy against measured hierarchical planning effort."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def _method(path: Path, split: str) -> str:
    marker = f"-{split}-k0_"
    if marker not in path.stem:
        raise ValueError(f"cannot recover method label from {path.name}")
    return path.stem.split(marker, 1)[0]


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.evaluation:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("rows"):
            raise ValueError(f"empty MPC evaluation: {path}")
        example = payload["rows"][0]
        effort = payload["mean_planning_wall_seconds"]
        rows.append({
            "method": _method(path, example["dataset_split"]),
            "dataset_split": example["dataset_split"],
            "k0": example["k0"], "k1": example["k1"],
            "planning_effort": effort,
            "planning_effort_unit": "measured_wall_seconds_per_episode",
            "accuracy": payload["accuracy"],
            "accuracy_ci95_low": payload["accuracy_ci95"][0],
            "accuracy_ci95_high": payload["accuracy_ci95"][1],
            "episodes": payload["episodes"],
            "generation_failure_rate": payload["generation_failure_rate"],
            "mean_manager_seconds": payload["mean_manager_seconds"],
            "mean_worker_seconds": payload["mean_worker_seconds"],
            "mean_exact_reencode_seconds": payload[
                "mean_exact_reencode_seconds"
            ],
            "mean_candidate_generation_seconds": payload[
                "mean_candidate_generation_seconds"
            ],
            "mean_candidate_exact_grounding_seconds": payload[
                "mean_candidate_exact_grounding_seconds"
            ],
            "mean_token_rollout_seconds": payload[
                "mean_token_rollout_seconds"
            ],
            "mean_worker_scoring_seconds": payload[
                "mean_worker_scoring_seconds"
            ],
            "mean_generated_candidate_tokens": payload[
                "mean_generated_candidate_tokens"
            ],
            "mean_exact_candidate_tensor_tokens": payload[
                "mean_exact_candidate_tensor_tokens"
            ],
            "mean_manager_transition_evaluations": payload[
                "mean_manager_transition_evaluations"
            ],
        })
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for planning plots") from error
    splits = sorted({row["dataset_split"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    figure, axes = plt.subplots(
        len(splits), 1, figsize=(9, 3.2 * len(splits)), squeeze=False
    )
    for axis, split in zip(axes[:, 0], splits):
        for method in methods:
            selected = sorted(
                (row for row in rows if row["dataset_split"] == split
                 and row["method"] == method),
                key=lambda row: row["planning_effort"],
            )
            if selected:
                x = [row["planning_effort"] for row in selected]
                y = [row["accuracy"] for row in selected]
                low = [row["accuracy_ci95_low"] for row in selected]
                high = [row["accuracy_ci95_high"] for row in selected]
                axis.plot(x, y, marker="o", label=method)
                axis.fill_between(x, low, high, alpha=0.10)
        axis.set_title(split)
        axis.set_xlabel("measured planning wall time per episode (seconds)")
        axis.set_ylabel("complete-solution accuracy")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=180)
    figure.savefig(args.output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
