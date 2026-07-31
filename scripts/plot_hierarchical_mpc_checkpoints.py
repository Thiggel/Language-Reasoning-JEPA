#!/usr/bin/env python3
"""Plot fixed-effort OOD accuracy over value-training checkpoints."""

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


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.evaluation:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload["checkpoint_metadata"]
        rows.append({
            "method": metadata["method"],
            "dataset_split": payload["rows"][0]["dataset_split"],
            "optimizer_step": metadata["optimizer_step"],
            "accuracy": payload["accuracy"],
            "episodes": payload["episodes"],
        })
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib.pyplot as plt
    splits = sorted({row["dataset_split"] for row in rows})
    figure, axes = plt.subplots(
        len(splits), 1, figsize=(8, 3 * len(splits)), squeeze=False
    )
    for axis, split in zip(axes[:, 0], splits):
        for method in sorted({row["method"] for row in rows}):
            selected = sorted(
                (row for row in rows if row["dataset_split"] == split
                 and row["method"] == method),
                key=lambda row: row["optimizer_step"],
            )
            if selected:
                axis.plot(
                    [row["optimizer_step"] for row in selected],
                    [row["accuracy"] for row in selected],
                    marker="o", label=method,
                )
        axis.set_title(split); axis.set_xlabel("value optimizer step")
        axis.set_ylabel("accuracy at K0=32, K1=2")
        axis.set_ylim(-0.02, 1.02); axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=180)
    figure.savefig(args.output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
