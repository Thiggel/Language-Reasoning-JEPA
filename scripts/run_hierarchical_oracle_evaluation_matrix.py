#!/usr/bin/env python3
"""Evaluate a trained token or nested JEPA through its admitted oracle gates."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import torch

from textjepa.training.hierarchical_language import ResearchStage
from textjepa.utils.language_planning_runtime import load_hierarchical_checkpoint


SPLITS = (
    "id_test",
    "near_length_ood",
    "far_length_ood",
    "structural_ood",
    "paraphrase_ood",
)
TOKEN_HORIZONS = (4, 8, 16)
WORKER_HORIZONS = (8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--method-label", required=True)
    parser.add_argument(
        "--metric", choices=("euclidean", "mahalanobis"), required=True
    )
    parser.add_argument("--max-roots", type=int, default=64)
    parser.add_argument("--flat-population", type=int, default=64)
    parser.add_argument("--worker-population", type=int, default=32)
    parser.add_argument("--greedy-examples", type=int, default=256)
    parser.add_argument("--include-greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def _run(command: list[str], records: list[dict]) -> None:
    started = perf_counter()
    subprocess.run(command, check=True)
    records.append({"command": command, "wall_seconds": perf_counter() - started})


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot(rows: list[dict], x: str, metrics: tuple[str, ...], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(metrics), figsize=(5.2 * len(metrics), 4.2), squeeze=False
    )
    for axis, metric in zip(axes[0], metrics):
        for split in SPLITS:
            selected = sorted(
                (row for row in rows if row["split"] == split),
                key=lambda row: row[x],
            )
            axis.plot(
                [row[x] for row in selected],
                [row[metric] for row in selected],
                marker="o", label=split,
            )
        axis.set(xlabel=x, ylabel=metric, title=metric.replace("_", " "))
        axis.grid(alpha=0.25)
    axes[0, -1].legend(fontsize=7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if min(
        args.max_roots, args.flat_population, args.worker_population,
        args.greedy_examples,
    ) < 1:
        raise ValueError("evaluation sizes must be positive")
    run_dir = os.environ.get("RUN_DIR")
    root = args.output_root or (Path(run_dir) / "evaluation" if run_dir else None)
    if root is None:
        raise ValueError("--output-root or RUN_DIR is required")
    root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for path in (
            args.source_root / f"features/{split}.pt",
            args.source_root / f"splits/{split}.jsonl",
        ):
            if not path.exists():
                raise FileNotFoundError(path)
    _, learner = load_hierarchical_checkpoint(args.checkpoint, "cpu")
    if learner.stage < ResearchStage.TOKEN_JEPA:
        raise ValueError("oracle evaluation requires a trained token JEPA")
    joint = learner.stage >= ResearchStage.SENTENCE_JEPA
    python = sys.executable
    commands: list[dict] = []
    flat_rows: list[dict] = []
    worker_rows: list[dict] = []
    greedy_rows: list[dict] = []
    for split in SPLITS:
        features = args.source_root / f"features/{split}.pt"
        examples = args.source_root / f"splits/{split}.jsonl"
        for horizon in TOKEN_HORIZONS:
            candidates = root / f"candidates/flat-{split}-k{horizon}.pt"
            metrics_path = root / f"metrics/flat-{split}-k{horizon}.json"
            _run([
                python, "scripts/build_flat_token_oracle_candidates.py",
                "--features", str(features), "--checkpoint", str(args.checkpoint),
                "--output", str(candidates), "--dataset-split", split,
                "--method-label", args.method_label, "--horizon", str(horizon),
                "--population", str(args.flat_population), "--max-roots",
                str(args.max_roots), "--seed", str(args.seed), "--device",
                args.device, "--dtype", args.dtype,
            ], commands)
            _run([
                python, "scripts/evaluate_hierarchical_language_oracles.py",
                "--checkpoint", str(args.checkpoint), "--candidates",
                str(candidates), "--output", str(metrics_path), "--mode",
                "flat_token", "--metric", args.metric, "--device", args.device,
            ], commands)
            payload = _read(metrics_path)
            flat_rows.append({
                "method": args.method_label, "split": split, "k": horizon,
                **payload["metrics"],
            })
        if joint:
            for horizon in WORKER_HORIZONS:
                candidates = root / f"candidates/worker-{split}-k0_{horizon}.pt"
                metrics_path = root / f"metrics/worker-{split}-k0_{horizon}.json"
                _run([
                    python, "scripts/build_sentence_waypoint_candidates.py",
                    "--features", str(features), "--examples", str(examples),
                    "--checkpoint", str(args.checkpoint), "--output",
                    str(candidates), "--dataset-split", split, "--k0",
                    str(horizon), "--population", str(args.worker_population),
                    "--max-roots", str(args.max_roots), "--seed", str(args.seed),
                    "--device", args.device, "--dtype", args.dtype,
                ], commands)
                validity = torch.load(
                    candidates, map_location="cpu", weights_only=True
                )["nested_validity"]
                row = {
                    "method": args.method_label, "split": split, "k0": horizon,
                    **validity["metrics"],
                    "nested_gate_passed": bool(validity["passed"]),
                }
                if validity["passed"]:
                    _run([
                        python, "scripts/evaluate_hierarchical_language_oracles.py",
                        "--checkpoint", str(args.checkpoint), "--candidates",
                        str(candidates), "--output", str(metrics_path), "--mode",
                        "sentence_worker", "--metric", args.metric, "--device",
                        args.device,
                    ], commands)
                    row.update(_read(metrics_path)["metrics"])
                else:
                    row.update({
                        "predicted_symbolic_waypoint_success": float("nan"),
                        "exact_symbolic_waypoint_success": float("nan"),
                        "latent_symbolic_disagreement": float("nan"),
                        "model_oracle_gap": float("nan"),
                    })
                worker_rows.append(row)
        if args.include_greedy:
            output = root / f"metrics/greedy-{split}.json"
            _run([
                python, "scripts/evaluate_hierarchical_language_controls.py",
                "--features", str(features), "--examples", str(examples),
                "--checkpoint", str(args.checkpoint), "--output", str(output),
                "--dataset-split", split, "--method-label", "frozen-qwen-greedy",
                "--mode", "greedy", "--metric", args.metric, "--k0", "64",
                "--k1", "1", "--max-examples", str(args.greedy_examples),
                "--seed", str(args.seed), "--device", args.device, "--dtype",
                args.dtype,
            ], commands)
            payload = _read(output)
            greedy_rows.append({
                "method": "frozen-qwen-greedy", "split": split,
                "accuracy": payload["accuracy"], "ci95_low": payload["accuracy_ci95"][0],
                "ci95_high": payload["accuracy_ci95"][1],
                "mean_wall_seconds": payload["mean_planning_wall_seconds"],
            })
    if joint:
        geometry = root / "geometry/id_test_sentence_geometry.pt"
        plot_dir = root / "figures/sentence_geometry"
        _run([
            python, "scripts/build_hierarchical_sentence_geometry.py",
            "--features", str(args.source_root / "features/id_test.pt"),
            "--examples", str(args.source_root / "splits/id_test.jsonl"),
            "--checkpoint", str(args.checkpoint), "--output", str(geometry),
            "--plot-output-dir", str(plot_dir), "--max-igsm-groups", "512",
            "--device", args.device, "--dtype", args.dtype,
        ], commands)
        # UMAP is optional in the shared cluster environment.  Preserve the
        # quantitative triplets and t-SNE even when that optional package is absent.
        attempted = subprocess.run([
            python, "scripts/plot_hierarchical_sentence_geometry.py",
            "--artifact", str(geometry), "--output-dir", str(plot_dir),
            "--method", "umap", "--seed", str(args.seed),
        ], check=False)
        commands.append({"command": "optional_umap", "returncode": attempted.returncode})
    _write_csv(root / "tables/flat_oracle_depth.csv", flat_rows)
    _plot(
        flat_rows, "k",
        ("predicted_next_token_accuracy", "exact_next_token_accuracy"),
        root / "figures/flat_oracle_depth.png",
    )
    if worker_rows:
        _write_csv(root / "tables/sentence_worker_depth.csv", worker_rows)
        _plot(
            worker_rows, "k0",
            ("predicted_symbolic_waypoint_success", "exact_symbolic_waypoint_success"),
            root / "figures/sentence_worker_depth.png",
        )
    if greedy_rows:
        _write_csv(root / "tables/frozen_qwen_greedy.csv", greedy_rows)
    (root / "workflow.json").write_text(json.dumps({
        "method": args.method_label,
        "checkpoint": str(args.checkpoint),
        "stage": learner.stage.name,
        "metric": args.metric,
        "splits": SPLITS,
        "token_horizons": TOKEN_HORIZONS,
        "worker_horizons": WORKER_HORIZONS if joint else [],
        "max_roots_per_cell": args.max_roots,
        "value_available": False,
        "candidate_privileged": True,
        "commands": commands,
    }, indent=2) + "\n", encoding="utf-8")
    (root / "outcome.json").write_text(json.dumps({
        "status": "complete", "method": args.method_label,
        "flat_oracle_evaluated": True,
        "sentence_worker_evaluated": joint,
        "value_evaluated": False,
        "reason_value_not_evaluated": "checkpoint has no Pi1 or V",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
