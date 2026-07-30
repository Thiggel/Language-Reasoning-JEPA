#!/usr/bin/env python3
"""Aggregate oracle evaluations into planning-effort/accuracy curves."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for path in args.evaluation:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("mode") != "flat_token":
            raise ValueError("planning-effort plot currently expects flat_token")
        for row in payload["rows"]:
            effort = int(row["planning_effort_candidate_tokens"])
            split = str(row["dataset_split"])
            depth = int(row["symbolic_depth"])
            grouped[(split, depth, effort)].append(row)
    rows = []
    metrics = (
        "predicted_span_accuracy", "exact_span_accuracy",
        "predicted_next_token_accuracy", "exact_next_token_accuracy",
    )
    for (split, depth, effort), values in sorted(grouped.items()):
        row = {
            "dataset_split": split,
            "symbolic_depth": depth,
            "planning_effort_candidate_tokens": effort,
            "roots": len(values),
            "planning_wall_seconds_mean": sum(
                value["planning_wall_seconds"] for value in values
            ) / len(values),
        }
        for metric in metrics:
            row[metric] = sum(value[metric] for value in values) / len(values)
        rows.append(row)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to render the plot") from error
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    splits = sorted({row["dataset_split"] for row in rows})
    colors = plt.get_cmap("tab10").colors
    for split_index, split in enumerate(splits):
        color = colors[split_index]
        split_rows = [row for row in rows if row["dataset_split"] == split]
        by_effort = defaultdict(list)
        for row in split_rows:
            by_effort[row["planning_effort_candidate_tokens"]].append(row)
        effort = sorted(by_effort)
        for axis, exact_name, predicted_name, title in (
            (
                axes[0], "exact_next_token_accuracy",
                "predicted_next_token_accuracy", "Next-token accuracy",
            ),
            (
                axes[1], "exact_span_accuracy",
                "predicted_span_accuracy", "Exact span accuracy",
            ),
        ):
            exact = [
                sum(item[exact_name] * item["roots"] for item in by_effort[x])
                / sum(item["roots"] for item in by_effort[x])
                for x in effort
            ]
            predicted = [
                sum(
                    item[predicted_name] * item["roots"]
                    for item in by_effort[x]
                ) / sum(item["roots"] for item in by_effort[x])
                for x in effort
            ]
            axis.plot(
                effort, exact, "--", marker="o", color=color,
                label=f"{split} exact",
            )
            axis.plot(
                effort, predicted, "-", marker="o", color=color,
                label=f"{split} learned",
            )
            axis.set_title(title)
            axis.set_xlabel("Planning effort (candidate token evaluations)")
            axis.set_ylabel("Accuracy")
            axis.grid(alpha=0.25)
    split_handles = [
        Line2D([0], [0], color=colors[index], marker="o", label=split)
        for index, split in enumerate(splits)
    ]
    kind_handles = [
        Line2D([0], [0], color="black", linestyle="--", label="Exact endpoint"),
        Line2D([0], [0], color="black", linestyle="-", label="Learned rollout"),
    ]
    figure.legend(
        handles=split_handles, loc="upper center", ncol=len(splits),
        bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=9,
    )
    axes[1].legend(
        handles=kind_handles, loc="lower left", frameon=True, fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=180)


if __name__ == "__main__":
    main()
