#!/usr/bin/env python3
"""Build fixed verified-depth iGSM manifests for language planning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from textjepa.data.language_planning import (
    IGSM_REFERENCE_COUNTS,
    IGSMSplit,
    balanced_igsm_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-out-graph-family", action="append", default=[])
    parser.add_argument("--held-out-template-family", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--count-scale", type=float, default=1.0,
        help="Use <1 only for explicit pilot manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.count_scale <= 1:
        raise ValueError("count-scale must lie in (0, 1]")
    with args.input.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if not records:
        raise ValueError("verified candidate pool is empty")
    graphs = set(args.held_out_graph_family)
    templates = set(args.held_out_template_family)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    seen_problem_ids: set[str] = set()
    for split, reference_count in IGSM_REFERENCE_COUNTS.items():
        count = max(1, round(reference_count * args.count_scale))
        selected = balanced_igsm_manifest(
            records, split, count,
            held_out_graph_families=graphs,
            held_out_template_families=templates,
            seed=args.seed,
        )
        path = args.output_dir / f"{split.value}.jsonl"
        selected_ids = [str(record["problem_id"]) for record in selected]
        overlap = seen_problem_ids.intersection(selected_ids)
        if overlap:
            raise ValueError(
                f"split manifests overlap on problem IDs: {sorted(overlap)[:3]}"
            )
        seen_problem_ids.update(selected_ids)
        with path.open("w", encoding="utf-8") as handle:
            for record in selected:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        summary[split.value] = {
            "count": len(selected),
            "reference_count": reference_count,
            "problem_id_fingerprint": hashlib.sha256(
                "\n".join(sorted(selected_ids)).encode()
            ).hexdigest(),
            "depth_histogram": {
                str(depth): sum(
                    int(record["reasoning_depth"]) == depth
                    for record in selected
                )
                for depth in sorted({
                    int(record["reasoning_depth"]) for record in selected
                })
            },
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps({
            "seed": args.seed,
            "count_scale": args.count_scale,
            "held_out_graph_families": sorted(graphs),
            "held_out_template_families": sorted(templates),
            "splits": summary,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
