#!/usr/bin/env python3
"""Plot fixed-depth OOD oracle accuracy over JEPA optimizer checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = defaultdict(list)
    for path in args.evaluation:
        match = re.search(r"step_(\d+)", path.name)
        if match is None:
            raise ValueError(f"cannot recover checkpoint step from {path}")
        step = int(match.group(1))
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            method = str(row.get("method_label", "token_jepa"))
            split = str(row["dataset_split"])
            grouped[(method, split, step)].append(row)
    rows = []
    for (method, split, step), values in sorted(grouped.items()):
        rows.append({
            "method": method,
            "dataset_split": split,
            "optimizer_step": step,
            "roots": len(values),
            "predicted_next_token_accuracy": sum(
                row["predicted_next_token_accuracy"] for row in values
            ) / len(values),
            "exact_next_token_accuracy": sum(
                row["exact_next_token_accuracy"] for row in values
            ) / len(values),
            "predicted_span_accuracy": sum(
                row["predicted_span_accuracy"] for row in values
            ) / len(values),
            "exact_span_accuracy": sum(
                row["exact_span_accuracy"] for row in values
            ) / len(values),
        })
    if not rows:
        raise ValueError("no checkpoint evaluation rows")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8"
    )
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    keys = sorted({(row["method"], row["dataset_split"]) for row in rows})
    colors = plt.get_cmap("tab20").colors
    for index, (method, split) in enumerate(keys):
        selected = sorted(
            (
                row for row in rows
                if row["method"] == method
                and row["dataset_split"] == split
            ),
            key=lambda row: row["optimizer_step"],
        )
        steps = [row["optimizer_step"] for row in selected]
        for axis, predicted, exact, title in (
            (
                axes[0], "predicted_next_token_accuracy",
                "exact_next_token_accuracy", "Next-token accuracy at k=8",
            ),
            (
                axes[1], "predicted_span_accuracy",
                "exact_span_accuracy", "Exact-span accuracy at k=8",
            ),
        ):
            axis.plot(
                steps, [row[predicted] for row in selected],
                marker="o", color=colors[index % len(colors)],
                label=f"{method} · {split}",
            )
            axis.plot(
                steps, [row[exact] for row in selected],
                linestyle="--", alpha=0.35,
                color=colors[index % len(colors)],
            )
            axis.set_title(title)
            axis.set_xlabel("Optimizer step")
            axis.set_ylabel("Accuracy")
            axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=180)


if __name__ == "__main__":
    main()
