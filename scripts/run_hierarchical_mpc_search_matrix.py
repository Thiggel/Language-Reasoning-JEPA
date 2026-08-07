#!/usr/bin/env python3
"""Run resumable oracle/value hierarchical-MPC search ablations.

This driver evaluates genuine token optimization rather than one-shot LM
reranking.  It records every cell separately, supports deterministic sharding,
and plots accuracy against both nominal horizons and measured model work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--mode", choices=("oracle", "value"), default="oracle")
    parser.add_argument(
        "--metric", choices=("euclidean", "mahalanobis"), required=True
    )
    parser.add_argument("--splits", nargs="+", default=["id_validation"])
    parser.add_argument(
        "--worker-search", nargs="+",
        choices=(
            "one_shot", "beam", "autoregressive_cem", "markov_cem",
            "factorized_cem",
        ), default=["one_shot", "beam", "autoregressive_cem", "markov_cem"],
    )
    parser.add_argument(
        "--worker-objectives", nargs="+",
        choices=("jepa", "combined", "lm"), default=["jepa"],
    )
    parser.add_argument(
        "--manager-support", nargs="+",
        choices=("ambient", "prior", "prior_trust", "prior_nll"),
        default=["prior_trust"],
    )
    parser.add_argument(
        "--execution", nargs="+", choices=("open_loop", "closed_loop"),
        default=["closed_loop"],
    )
    parser.add_argument("--k0", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--k1", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--worker-population", type=int, default=64)
    parser.add_argument("--worker-iterations", type=int, default=4)
    parser.add_argument("--worker-elite-fraction", type=float, default=0.1)
    parser.add_argument("--worker-beam-width", type=int, default=8)
    parser.add_argument("--worker-branch-factor", type=int, default=8)
    parser.add_argument("--worker-preserve-prefix", type=int, default=4)
    parser.add_argument(
        "--worker-execution-tokens", nargs="+", type=int, default=[0]
    )
    parser.add_argument("--manager-population", type=int, default=256)
    parser.add_argument("--manager-iterations", type=int, default=4)
    parser.add_argument("--manager-elite-fraction", type=float, default=0.1)
    parser.add_argument("--manager-trust-region", type=float, default=3.0)
    parser.add_argument("--max-reasoning-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--worker-prior-weight", type=float, default=0.01)
    parser.add_argument("--manager-prior-weight", type=float, default=0.1)
    parser.add_argument("--include-geometry", action="store_true")
    parser.add_argument("--geometry-groups", type=int, default=256)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def _cell_id(cell: dict) -> str:
    return "-".join([
        cell["split"], cell["worker_search"], cell["worker_objective"],
        cell["manager_support"], cell["execution"],
        f'nexec_{cell["worker_execution_tokens"]}',
        f'k0_{cell["k0"]}', f'k1_{cell["k1"]}',
    ])


def _cells(args) -> list[dict]:
    cells = []
    for split in args.splits:
        for worker in args.worker_search:
            for worker_objective in args.worker_objectives:
                if worker in {"markov_cem", "factorized_cem"} and (
                    worker_objective != "jepa"
                ):
                    continue
                for manager in args.manager_support:
                    for execution in args.execution:
                        for execution_tokens in args.worker_execution_tokens:
                            if worker == "one_shot" and execution_tokens:
                                continue
                            for k0 in args.k0:
                                for k1 in args.k1:
                                    cell = {
                                        "split": split,
                                        "worker_search": worker,
                                        "worker_objective": worker_objective,
                                        "manager_support": manager,
                                        "execution": execution,
                                        "worker_execution_tokens": (
                                            execution_tokens
                                        ),
                                        "k0": k0, "k1": k1,
                                    }
                                    digest = int(hashlib.sha256(
                                        _cell_id(cell).encode()
                                    ).hexdigest(), 16)
                                    if (
                                        digest % args.shard_count
                                        == args.shard_index
                                    ):
                                        cells.append(cell)
    return cells


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = sorted({
        f'{row["worker_search"]}/{row["manager_support"]}/'
        f'{row["execution"]}/n={row["worker_execution_tokens"]}'
        for row in rows
    })
    for label in labels:
        selected = [row for row in rows if (
            f'{row["worker_search"]}/{row["manager_support"]}/'
            f'{row["execution"]}/n={row["worker_execution_tokens"]}' == label
        )]
        selected.sort(key=lambda row: row["mean_transition_evaluations"])
        axes[0].plot(
            [row["mean_transition_evaluations"] for row in selected],
            [row["accuracy"] for row in selected], marker="o", label=label,
        )
        selected.sort(key=lambda row: (row["k0"], row["k1"]))
        axes[1].scatter(
            [row["k0"] for row in selected],
            [row["accuracy"] for row in selected],
            s=[25 + 12 * row["k1"] for row in selected], label=label,
        )
        axes[2].scatter(
            [row["mean_planning_wall_seconds"] for row in selected],
            [row["accuracy"] for row in selected], label=label,
        )
    axes[0].set(xlabel="mean JEPA transition evaluations", ylabel="accuracy")
    axes[1].set(xlabel="K0 token horizon", ylabel="accuracy",
                title="marker size increases with K1")
    axes[2].set(xlabel="mean planning wall seconds", ylabel="accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[2].legend(fontsize=6, loc="best")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    sizes = (
        args.max_examples, args.worker_population, args.worker_iterations,
        args.worker_beam_width, args.worker_branch_factor,
        args.manager_population, args.manager_iterations,
        args.max_reasoning_steps, args.geometry_groups,
    )
    if min(sizes) < 1:
        raise ValueError("matrix sizes must be positive")
    root_env = os.environ.get("RUN_DIR")
    root = args.output_root or (Path(root_env) if root_env else None)
    if root is None:
        raise ValueError("--output-root or RUN_DIR is required")
    root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    rows, commands = [], []
    for cell in _cells(args):
        features = args.source_root / "features" / f'{cell["split"]}.pt'
        examples = args.source_root / "splits" / f'{cell["split"]}.jsonl'
        if not features.exists() or not examples.exists():
            raise FileNotFoundError(f"missing split artifacts for {cell['split']}")
        output = root / "cells" / f"{_cell_id(cell)}.json"
        command = [
            python, "scripts/evaluate_hierarchical_language_mpc.py",
            "--features", str(features), "--examples", str(examples),
            "--checkpoint", str(args.checkpoint), "--output", str(output),
            "--dataset-split", cell["split"], "--method-label",
            args.method_label, "--mode", args.mode, "--metric", args.metric,
            "--k0", str(cell["k0"]), "--k1", str(cell["k1"]),
            "--worker-search", cell["worker_search"],
            "--worker-objective", cell["worker_objective"],
            "--manager-action-support", cell["manager_support"],
            "--manager-grounding", "none",
            "--hierarchy-execution", cell["execution"],
            "--worker-population", str(args.worker_population),
            "--worker-iterations", str(args.worker_iterations),
            "--worker-elite-fraction", str(args.worker_elite_fraction),
            "--worker-beam-width", str(args.worker_beam_width),
            "--worker-branch-factor", str(args.worker_branch_factor),
            "--worker-preserve-prefix", str(args.worker_preserve_prefix),
            "--worker-execution-tokens",
            str(cell["worker_execution_tokens"]),
            "--worker-prior-weight", str(args.worker_prior_weight),
            "--manager-population", str(args.manager_population),
            "--cem-iterations", str(args.manager_iterations),
            "--elite-fraction", str(args.manager_elite_fraction),
            "--manager-trust-region", str(args.manager_trust_region),
            "--prior-weight", str(args.manager_prior_weight),
            "--max-examples", str(args.max_examples),
            "--max-reasoning-steps", str(args.max_reasoning_steps),
            "--temperature", str(args.temperature), "--top-p", str(args.top_p),
            "--top-k", str(args.top_k), "--seed", str(args.seed),
            "--device", args.device, "--dtype", args.dtype,
        ]
        started = perf_counter()
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
        commands.append({
            "cell": _cell_id(cell), "command": command,
            "wall_seconds": perf_counter() - started,
        })
        payload = json.loads(output.read_text(encoding="utf-8"))
        rows.append({
            **cell, "method": args.method_label,
            "accuracy": payload["accuracy"],
            "ci95_low": payload["accuracy_ci95"][0],
            "ci95_high": payload["accuracy_ci95"][1],
            "episodes": payload["episodes"],
            "failure_rate": payload["generation_failure_rate"],
            "mean_planning_wall_seconds": payload["mean_planning_wall_seconds"],
            "mean_token_transition_evaluations": payload[
                "mean_token_transition_evaluations"
            ],
            "mean_manager_transition_evaluations": payload[
                "mean_manager_transition_evaluations"
            ],
            "mean_transition_evaluations": (
                payload["mean_token_transition_evaluations"]
                + payload["mean_manager_transition_evaluations"]
            ),
            "mean_manager_replans": payload["mean_manager_replans"],
            "mean_worker_replans": payload["mean_worker_replans"],
            "mean_worker_seconds": payload["mean_worker_seconds"],
            "mean_manager_seconds": payload["mean_manager_seconds"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "dataset_fingerprint": payload["dataset_fingerprint"],
        })
        _write_csv(root / "tables" / "mpc_search_matrix.partial.csv", rows)
    _write_csv(root / "tables" / "mpc_search_matrix.csv", rows)
    _plot(rows, root / "figures" / "accuracy_vs_planning_effort.png")
    if args.include_geometry and args.shard_index == 0:
        split = "id_test" if "id_test" in args.splits else args.splits[0]
        geometry = root / "geometry" / f"{split}_sentence_geometry.pt"
        geometry.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            python, "scripts/build_hierarchical_sentence_geometry.py",
            "--features", str(args.source_root / "features" / f"{split}.pt"),
            "--examples", str(args.source_root / "splits" / f"{split}.jsonl"),
            "--checkpoint", str(args.checkpoint), "--output", str(geometry),
            "--plot-output-dir", str(root / "figures" / "sentence_geometry"),
            "--max-igsm-groups", str(args.geometry_groups),
            "--device", args.device, "--dtype", args.dtype,
        ], check=True)
        subprocess.run([
            python, "scripts/plot_hierarchical_sentence_geometry.py",
            "--artifact", str(geometry), "--output-dir",
            str(root / "figures" / "sentence_geometry"),
            "--method", "umap", "--seed", str(args.seed),
        ], check=False)
    (root / "workflow.json").write_text(json.dumps({
        "method": args.method_label, "mode": args.mode,
        "metric": args.metric, "checkpoint": str(args.checkpoint),
        "source_root": str(args.source_root),
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "cells": [_cell_id(cell) for cell in _cells(args)],
        "candidate_privileged": args.mode == "oracle",
        "terminal_available_to_model": False,
        "terminal_available_to_planner": args.mode == "oracle",
        "commands": commands,
    }, indent=2) + "\n", encoding="utf-8")
    (root / "outcome.json").write_text(json.dumps({
        "status": "complete", "cells": len(rows),
        "best_accuracy": max((row["accuracy"] for row in rows), default=None),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
