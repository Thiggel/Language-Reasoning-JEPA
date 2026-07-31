#!/usr/bin/env python3
"""Run the bounded Stage 0–2 / first-five-mechanism iGSM pilot."""

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count-scale", type=float, default=0.001)
    parser.add_argument("--pool-per-depth-family", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--checkpoint-step", type=int, action="append", default=[])
    parser.add_argument("--max-replay-steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--replay-batch-size", type=int, default=32)
    parser.add_argument("--multistep-replay-batch-size", type=int, default=4)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--max-eval-roots", type=int, default=16)
    parser.add_argument("--eval-all-checkpoints", action="store_true")
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--reencode-batch-size", type=int, default=32)
    parser.add_argument("--counterfactual-example-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def replay_batch_size(variant: str, args: argparse.Namespace) -> int:
    """Use a smaller activation microbatch for recursive rollout training."""
    return (
        args.multistep_replay_batch_size
        if variant == "multistep"
        else args.replay_batch_size
    )


def _run(command: list[str], log: list[dict]) -> None:
    started = perf_counter()
    subprocess.run(command, check=True)
    log.append({
        "command": command,
        "wall_seconds": perf_counter() - started,
    })


def main() -> None:
    args = parse_args()
    if args.replay_batch_size <= 0 or args.multistep_replay_batch_size <= 0:
        raise ValueError("replay microbatch sizes must be positive")
    root = args.output_root or Path(os.environ.get("RUN_DIR", ""))
    if not str(root):
        raise ValueError("--output-root or RUN_DIR is required")
    root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    commands: list[dict] = []
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
    feature_paths = {}
    for split in ("train", *EVAL_SPLITS):
        feature = root / "features" / f"{split}.pt"
        command = [
            python, "scripts/collect_hierarchical_language_features.py",
            "--input", str(splits / f"{split}.jsonl"),
            "--output", str(feature),
            "--device", args.device, "--dtype", args.dtype,
            "--seed", str(args.seed), "--resume",
            "--generation-batch-size", str(args.generation_batch_size),
            "--reencode-batch-size", str(args.reencode_batch_size),
        ]
        if split == "train":
            command.extend([
                "--counterfactual-output",
                str(root / "features" / "train_counterfactual.pt"),
                "--counterfactual-example-limit",
                str(args.counterfactual_example_limit),
            ])
        _run(command, commands)
        feature_paths[split] = feature
    stage0 = root / "stage0_validation"
    _run([
        python, "scripts/train_hierarchical_language_jepa.py",
        "--features", str(feature_paths["train"]),
        "--output", str(stage0),
        "--experiment-config",
        "configs/experiment/hierarchical_language_data_validation.yaml",
        "--epochs", "1", "--device", args.device,
    ], commands)
    variants = {
        "dense": "hierarchical_language_token_jepa.yaml",
        "counterfactual": "hierarchical_language_token_counterfactual.yaml",
        "multistep": "hierarchical_language_token_multistep.yaml",
    }
    checkpoints = {}
    for name, config in variants.items():
        output = root / "models" / name
        command = [
            python, "scripts/train_hierarchical_language_jepa.py",
            "--features", str(feature_paths["train"]),
            "--output", str(output),
            "--experiment-config", f"configs/experiment/{config}",
            "--epochs", str(args.epochs), "--batch-size", "8",
            "--replay-batch-size", str(replay_batch_size(name, args)),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--max-optimizer-steps", str(args.max_optimizer_steps),
            "--warmup-steps", str(args.warmup_steps),
            "--max-replay-steps-per-epoch",
            str(args.max_replay_steps_per_epoch),
            "--seed", str(args.seed), "--device", args.device,
        ]
        for checkpoint_step in args.checkpoint_step:
            command.extend(["--checkpoint-step", str(checkpoint_step)])
        if name != "dense":
            command.extend([
                "--counterfactual-features",
                str(root / "features" / "train_counterfactual.pt"),
            ])
        _run(command, commands)
        checkpoints[name] = output / "model.pt"
    evaluations = []
    # Full ID/OOD effort curves for every final JEPA variant.
    for variant in variants:
      for split in EVAL_SPLITS:
        for horizon in (4, 8, 16):
            candidate = (
                root / "candidates" /
                f"{variant}-{split}-k{horizon}.pt"
            )
            _run([
                python, "scripts/build_flat_token_oracle_candidates.py",
                "--features", str(feature_paths[split]),
                "--checkpoint", str(checkpoints[variant]),
                "--output", str(candidate),
                "--dataset-split", split,
                "--method-label", variant,
                "--horizon", str(horizon),
                "--population", str(args.population),
                "--max-roots", str(args.max_eval_roots),
                "--seed", str(args.seed),
                "--device", args.device, "--dtype", args.dtype,
            ], commands)
            for metric in (
                "mahalanobis", "euclidean", "cosine",
                "normalized_euclidean",
            ) if (
                variant == "dense" and split == "id_test" and horizon == 8
            ) else ("mahalanobis",):
                evaluation = (
                    root / "evaluations" /
                    f"{variant}-{split}-k{horizon}-{metric}.json"
                )
                _run([
                    python, "scripts/evaluate_hierarchical_language_oracles.py",
                    "--checkpoint", str(checkpoints[variant]),
                    "--candidates", str(candidate),
                    "--output", str(evaluation),
                    "--mode", "flat_token", "--metric", metric,
                    "--device", args.device,
                ], commands)
                if metric == "mahalanobis":
                    evaluations.append(evaluation)
    _run([
        python, "scripts/plot_hierarchical_planning_effort.py",
        *sum((["--evaluation", str(path)] for path in evaluations), []),
        "--output-prefix", str(root / "figures" / "planning_effort_accuracy"),
    ], commands)
    checkpoint_evaluations = []
    if args.eval_all_checkpoints:
        for variant in variants:
            checkpoint_dir = root / "models" / variant / "checkpoints"
            for checkpoint in sorted(checkpoint_dir.glob("step_*.pt")):
                for split in EVAL_SPLITS[1:]:
                    stem = f"{variant}-{checkpoint.stem}-{split}-k8"
                    candidate = root / "checkpoint_candidates" / f"{stem}.pt"
                    evaluation = (
                        root / "checkpoint_evaluations" / f"{stem}.json"
                    )
                    _run([
                        python, "scripts/build_flat_token_oracle_candidates.py",
                        "--features", str(feature_paths[split]),
                        "--checkpoint", str(checkpoint),
                        "--output", str(candidate),
                        "--dataset-split", split, "--horizon", "8",
                        "--method-label", variant,
                        "--population", str(args.population),
                        "--max-roots", str(args.max_eval_roots),
                        "--seed", str(args.seed),
                        "--device", args.device, "--dtype", args.dtype,
                    ], commands)
                    _run([
                        python,
                        "scripts/evaluate_hierarchical_language_oracles.py",
                        "--checkpoint", str(checkpoint),
                        "--candidates", str(candidate),
                        "--output", str(evaluation),
                        "--mode", "flat_token", "--metric", "mahalanobis",
                        "--device", args.device,
                    ], commands)
                    checkpoint_evaluations.append(evaluation)
        if checkpoint_evaluations:
            _run([
                python, "scripts/plot_hierarchical_checkpoint_curves.py",
                *sum((
                    ["--evaluation", str(path)]
                    for path in checkpoint_evaluations
                ), []),
                "--output-prefix",
                str(root / "figures" / "ood_accuracy_over_training"),
            ], commands)
    summary = {
        "status": "completed",
        "seed": args.seed,
        "decision": (
            "Does token JEPA preserve exact-oracle gains as token planning "
            "effort increases, and do counterfactual/multistep additions "
            "reduce the exact-versus-predicted planning gap?"
        ),
        "candidate_privileged": True,
        "symbolic_targets_used_for_model_training": False,
        "splits": list(EVAL_SPLITS),
        "horizons": [4, 8, 16],
        "optimizer_steps": args.max_optimizer_steps,
        "checkpoint_steps": args.checkpoint_step,
        "population": args.population,
        "commands": commands,
        "artifacts": {
            "plot": "figures/planning_effort_accuracy.png",
            "table": "figures/planning_effort_accuracy.csv",
            "evaluations": "evaluations/",
            "models": "models/",
        },
    }
    (root / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
