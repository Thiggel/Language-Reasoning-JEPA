#!/usr/bin/env python3
"""Build matched iGSM data and run one parameter-matched Qwen control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


EVAL_SPLITS = (
    "id_test", "near_length_ood", "far_length_ood",
    "structural_ood", "paraphrase_ood",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--variant", choices=("added_capacity", "unfrozen"), required=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count-scale", type=float, default=0.05)
    parser.add_argument("--pool-per-depth-family", type=int, default=5_000)
    parser.add_argument("--steps", type=int, default=6_000)
    parser.add_argument("--max-eval-examples", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def _run(command: list[str], log: list[dict]) -> None:
    started = perf_counter()
    subprocess.run(command, check=True)
    log.append({"command": command, "wall_seconds": perf_counter() - started})


def main() -> None:
    args = parse_args()
    root = args.output_root or Path(os.environ.get("RUN_DIR", ""))
    if not str(root):
        raise ValueError("--output-root or RUN_DIR is required")
    root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    commands = []
    pool = root / "igsm_pool.jsonl"
    splits = root / "splits"
    _run([
        python, "scripts/generate_hierarchical_igsm_pool.py",
        "--output", str(pool),
        "--per-depth-family", str(args.pool_per_depth_family),
        "--seed", str(args.seed),
    ], commands)
    _run([
        python, "scripts/build_hierarchical_igsm_splits.py",
        "--input", str(pool), "--output-dir", str(splits),
        "--held-out-graph-family", "query_op_mul",
        "--held-out-template-family", "paraphrase_v1",
        "--count-scale", str(args.count_scale),
        "--seed", str(args.seed),
    ], commands)
    token_paths = {}
    for split in ("train", *EVAL_SPLITS):
        path = root / "tokens" / f"{split}.pt"
        _run([
            python, "scripts/collect_hierarchical_language_tokens.py",
            "--input", str(splits / f"{split}.jsonl"),
            "--output", str(path),
        ], commands)
        token_paths[split] = path
    command = [
        python, "scripts/train_parameter_matched_qwen_igsm.py",
        "--features", str(token_paths["train"]),
        "--split-jsonl", str(splits / "train.jsonl"),
        "--output", str(root / "model"),
        "--variant", args.variant,
        "--steps", str(args.steps),
        "--max-eval-examples", str(args.max_eval_examples),
        "--seed", str(args.seed),
        "--device", args.device,
        "--dtype", args.dtype,
    ]
    for split in EVAL_SPLITS:
        command.extend([
            "--eval-features", str(token_paths[split]),
            "--eval-jsonl", str(splits / f"{split}.jsonl"),
        ])
    _run(command, commands)
    (root / "run_summary.json").write_text(json.dumps({
        "status": "completed",
        "variant": args.variant,
        "seed": args.seed,
        "steps": args.steps,
        "count_scale": args.count_scale,
        "candidate_privileged": False,
        "symbolic_targets_used_for_model_training": False,
        "commands": commands,
        "artifacts": {
            "metrics": "model/metrics.json",
            "checkpoints": "model/checkpoints/",
        },
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
