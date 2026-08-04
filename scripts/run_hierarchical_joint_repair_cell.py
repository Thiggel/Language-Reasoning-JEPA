#!/usr/bin/env python3
"""Train and audit one matched token/joint collapse-repair cell."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 3 or args.learning_rate <= 0:
        raise ValueError("repair training scale is invalid")
    train_features = args.source_root / "features/train.pt"
    replay = args.source_root / "features/train_counterfactual.pt"
    validation = args.source_root / "features/id_validation.pt"
    for path in (train_features, replay, validation):
        if not path.is_file():
            raise ValueError(f"required source artifact is missing: {path}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_root / "model"
    checkpoints = sorted({
        max(1, args.steps // 10),
        max(1, args.steps // 3),
        max(1, 2 * args.steps // 3),
        args.steps,
    })
    command = [
        sys.executable,
        "scripts/train_hierarchical_language_jepa.py",
        "--features", str(train_features),
        "--counterfactual-features", str(replay),
        "--output", str(model_dir),
        "--experiment-config", f"configs/experiment/{args.config}",
        "--epochs", "100000",
        "--batch-size", "8",
        "--replay-batch-size", "16",
        "--max-optimizer-steps", str(args.steps),
        "--warmup-steps", str(max(100, round(args.steps * 0.05))),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", "0.01",
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    for step in checkpoints:
        command.extend(["--checkpoint-step", str(step)])
    _run(command)
    rows = []
    for step in checkpoints:
        checkpoint = model_dir / "checkpoints" / f"step_{step:06d}.pt"
        health = args.output_root / "health" / f"step_{step:06d}.json"
        _run([
            sys.executable,
            "scripts/evaluate_hierarchical_joint_health.py",
            "--features", str(validation),
            "--checkpoint", str(checkpoint),
            "--output", str(health),
            "--max-examples", "256",
            "--device", args.device,
        ])
        payload = json.loads(health.read_text())
        rows.append({"step": step, **payload["metrics"]})
    csv_path = args.output_root / "health_over_time.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[-1]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        [row["step"] for row in rows],
        [row["token_participation_ratio"] for row in rows],
        marker="o", label="token",
    )
    axes[1].plot(
        [row["step"] for row in rows],
        [row["token_identity_to_full"] for row in rows],
        marker="o", label="token",
    )
    if "sentence_participation_ratio" in rows[-1]:
        axes[0].plot(
            [row["step"] for row in rows],
            [row["sentence_participation_ratio"] for row in rows],
            marker="o", label="sentence",
        )
        axes[1].plot(
            [row["step"] for row in rows],
            [row["sentence_identity_to_full"] for row in rows],
            marker="o", label="sentence",
        )
    axes[0].set(xlabel="optimizer steps", ylabel="participation ratio")
    axes[1].set(xlabel="optimizer steps", ylabel="identity error / model error")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(args.label)
    figure.tight_layout()
    figure.savefig(args.output_root / "health_over_time.png", dpi=180)
    plt.close(figure)
    final = rows[-1]
    passed = (
        final["token_identity_to_full"] > 1.05
        and final["token_action_only_to_full"] > 1.05
        and final["token_cache_only_to_full"] > 1.05
        and final["token_participation_ratio"] >= 16.0
    )
    joint = "sentence_identity_to_full" in final
    if joint:
        passed = passed and (
            final["sentence_identity_to_full"] > 1.05
            and final["sentence_action_only_to_full"] > 1.05
            and final["sentence_cache_only_to_full"] > 1.05
            and final["sentence_participation_ratio"] >= 4.0
        )
    outcome = {
        "status": "completed" if passed else "validity_gate_stop",
        "passed": passed,
        "label": args.label,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "joint_token_sentence": joint,
        "final_metrics": final,
        "health_csv": str(csv_path),
    }
    (args.output_root / "outcome.json").write_text(
        json.dumps(outcome, indent=2) + "\n"
    )
    print(json.dumps(outcome), flush=True)


if __name__ == "__main__":
    main()
