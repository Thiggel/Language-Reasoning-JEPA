#!/usr/bin/env python3
"""Prepare the canonical 5% iGSM source then run the full hierarchy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


SPLITS = (
    "train", "id_validation", "id_test", "near_length_ood",
    "far_length_ood", "structural_ood", "paraphrase_ood",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    root = args.output_root or Path(os.environ.get("RUN_DIR", ""))
    if not str(root):
        raise ValueError("--output-root or RUN_DIR is required")
    root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    source = root / "source"
    pool = source / "igsm_pool.jsonl"
    splits = source / "splits"
    _run([
        python, "scripts/generate_hierarchical_igsm_pool.py",
        "--output", str(pool), "--per-depth-family", "5000",
        "--seed", str(args.seed),
    ])
    _run([
        python, "scripts/build_hierarchical_igsm_splits.py",
        "--input", str(pool), "--output-dir", str(splits),
        "--held-out-graph-family", "query_op_mul",
        "--held-out-template-family", "paraphrase_v1",
        "--count-scale", "0.05", "--seed", str(args.seed),
    ])
    for split in SPLITS:
        output = source / "features" / f"{split}.pt"
        command = [
            python, "scripts/collect_hierarchical_language_features.py",
            "--input", str(splits / f"{split}.jsonl"),
            "--output", str(output), "--device", args.device,
            "--dtype", args.dtype, "--seed", str(args.seed), "--resume",
            "--generation-batch-size", "16", "--reencode-batch-size", "32",
        ]
        if split == "train":
            command.extend([
                "--counterfactual-output",
                str(source / "features/train_counterfactual.pt"),
                "--counterfactual-example-limit", "2000",
            ])
        _run(command)
    full = root / "full"
    _run([
        python, "scripts/run_full_hierarchical_language_experiment.py",
        "--source-root", str(source), "--output-root", str(full),
        "--seed", str(args.seed),
        "--token-steps", "60000", "--sentence-steps", "60000",
        "--commutation-steps", "20000", "--macro-steps", "60000",
        "--value-steps", "30000", "--value-roots", "2048",
        "--eval-examples", "16", "--device", args.device,
        "--dtype", args.dtype,
    ])
    nested_outcome = full / "outcome.json"
    if not nested_outcome.exists():
        raise RuntimeError("full hierarchy did not write its outcome")
    shutil.copyfile(nested_outcome, root / "outcome.json")


if __name__ == "__main__":
    main()
