#!/usr/bin/env python3
"""Run the staged strict-nested iGSM hierarchy and its 2x2 ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import torch


SPLITS = (
    "id_test", "near_length_ood", "far_length_ood",
    "structural_ood", "paraphrase_ood",
)
DEPTH_PAIRS = (
    (8, 2), (16, 2), (32, 1), (32, 2), (32, 4), (32, 8), (64, 2),
)


class ValidityGateFailure(RuntimeError):
    """A scientifically valid early stop, distinct from a runtime failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--token-steps", type=int, default=60000)
    parser.add_argument("--sentence-steps", type=int, default=60000)
    parser.add_argument("--commutation-steps", type=int, default=20000)
    parser.add_argument("--macro-steps", type=int, default=60000)
    parser.add_argument("--value-steps", type=int, default=30000)
    parser.add_argument("--value-roots", type=int, default=2048)
    parser.add_argument("--eval-examples", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def _run(command: list[str], log: list[dict]) -> None:
    started = perf_counter()
    subprocess.run(command, check=True)
    log.append({"command": command, "wall_seconds": perf_counter() - started})


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _admit(
    checkpoint: Path,
    evaluation_dataset_fingerprint: str,
    admitted_stage: str,
    metrics: dict[str, float],
    passed: bool,
    output: Path,
) -> None:
    if any(not math.isfinite(float(value)) for value in metrics.values()):
        raise ValidityGateFailure(
            f"non-finite {admitted_stage} gate metrics"
        )
    record = {
        "passed": bool(passed),
        "admitted_stage": admitted_stage,
        "metrics": {name: float(value) for name, value in metrics.items()},
        "dataset_fingerprint": evaluation_dataset_fingerprint,
        "evaluation_dataset_fingerprint": evaluation_dataset_fingerprint,
        "admitted_checkpoint_sha256": _sha(checkpoint),
    }
    _write_json(output, record)
    if not passed:
        raise ValidityGateFailure(
            f"{admitted_stage} validity gate failed: {metrics}"
        )


def _train(
    python: str,
    commands: list[dict],
    *,
    features: Path,
    counterfactual: Path | None,
    output: Path,
    config: str,
    steps: int,
    learning_rate: float,
    seed: int,
    device: str,
    init: Path | None = None,
    admission: Path | None = None,
    value_replay: Path | None = None,
) -> None:
    command = [
        python, "scripts/train_hierarchical_language_jepa.py",
        "--features", str(features), "--output", str(output),
        "--experiment-config", f"configs/experiment/{config}",
        "--epochs", "100000", "--batch-size", "8",
        "--replay-batch-size", "16",
        "--max-optimizer-steps", str(steps),
        "--warmup-steps", str(max(100, round(steps * 0.05))),
        "--checkpoint-step", str(max(1, steps // 3)),
        "--checkpoint-step", str(max(1, 2 * steps // 3)),
        "--checkpoint-step", str(steps),
        "--learning-rate", str(learning_rate),
        "--weight-decay", "0.01", "--seed", str(seed),
        "--device", device,
    ]
    if counterfactual is not None:
        command.extend(["--counterfactual-features", str(counterfactual)])
    if init is not None:
        command.extend(["--init-checkpoint", str(init)])
    if admission is not None:
        command.extend(["--admission", str(admission)])
    if value_replay is not None:
        command.extend(["--value-replay", str(value_replay)])
    _run(command, commands)


def main() -> None:
    args = parse_args()
    if min(
        args.token_steps, args.sentence_steps, args.commutation_steps,
        args.macro_steps, args.value_steps, args.value_roots,
        args.eval_examples,
    ) < 1:
        raise ValueError("experiment scales must be positive")
    root = args.output_root or Path(os.environ.get("RUN_DIR", ""))
    if not str(root):
        raise ValueError("--output-root or RUN_DIR is required")
    root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    commands: list[dict] = []
    train_features = args.source_root / "features/train.pt"
    counterfactual = args.source_root / "features/train_counterfactual.pt"
    for path in (train_features, counterfactual):
        if not path.exists():
            raise FileNotFoundError(path)
    validation_features = args.source_root / "features/id_validation.pt"
    if not validation_features.exists():
        validation_features = root / "features/id_validation.pt"
        _run([
            python, "scripts/collect_hierarchical_language_features.py",
            "--input", str(args.source_root / "splits/id_validation.jsonl"),
            "--output", str(validation_features), "--device", args.device,
            "--dtype", args.dtype, "--seed", str(args.seed), "--resume",
        ], commands)
    token_dir = root / "models/token"
    _train(
        python, commands, features=train_features,
        counterfactual=counterfactual, output=token_dir,
        config="hierarchical_language_token_multistep.yaml",
        steps=args.token_steps, learning_rate=1e-3,
        seed=args.seed, device=args.device,
    )
    token_checkpoint = token_dir / "model.pt"
    flat_candidates = root / "gates/flat_candidates.pt"
    flat_metrics_path = root / "gates/flat_metrics.json"
    _run([
        python, "scripts/build_flat_token_oracle_candidates.py",
        "--features", str(validation_features),
        "--checkpoint", str(token_checkpoint),
        "--output", str(flat_candidates), "--dataset-split", "id_validation",
        "--method-label", "full-token", "--horizon", "8",
        "--population", "64", "--max-roots", "64",
        "--seed", str(args.seed), "--device", args.device,
        "--dtype", args.dtype,
    ], commands)
    _run([
        python, "scripts/evaluate_hierarchical_language_oracles.py",
        "--checkpoint", str(token_checkpoint),
        "--candidates", str(flat_candidates),
        "--output", str(flat_metrics_path), "--mode", "flat_token",
        "--metric", "mahalanobis", "--device", args.device,
    ], commands)
    flat_payload = _json(flat_metrics_path)
    flat = flat_payload["metrics"]
    flat["relative_model_gap"] = flat["model_oracle_gap"] / max(
        flat["exact_oracle_gain"], 1e-8
    )
    flat_admission = root / "gates/flat_admission.json"
    _admit(
        token_checkpoint, flat_payload["dataset_fingerprint"],
        "FLAT_ORACLE_TOKEN", flat,
        flat["exact_oracle_gain"] > 0
        and flat["relative_model_gap"] <= 1.0,
        flat_admission,
    )
    sentence_dir = root / "models/sentence"
    _train(
        python, commands, features=train_features,
        counterfactual=counterfactual, output=sentence_dir,
        config="hierarchical_language_nested_sentence.yaml",
        steps=args.sentence_steps, learning_rate=1e-3,
        seed=args.seed, device=args.device,
        init=token_checkpoint, admission=flat_admission,
    )
    sentence_checkpoint = sentence_dir / "model.pt"
    worker_candidates = root / "gates/sentence_worker_candidates.pt"
    worker_metrics_path = root / "gates/sentence_worker_metrics.json"
    _run([
        python, "scripts/build_sentence_waypoint_candidates.py",
        "--features", str(validation_features), "--examples",
        str(args.source_root / "splits/id_validation.jsonl"),
        "--checkpoint", str(sentence_checkpoint), "--output",
        str(worker_candidates), "--dataset-split", "id_validation",
        "--k0", "64", "--population", "32", "--max-roots", "64",
        "--seed", str(args.seed), "--device", args.device,
        "--dtype", args.dtype,
    ], commands)
    _run([
        python, "scripts/evaluate_hierarchical_language_oracles.py",
        "--checkpoint", str(sentence_checkpoint), "--candidates",
        str(worker_candidates), "--output", str(worker_metrics_path),
        "--mode", "sentence_worker", "--metric", "mahalanobis",
        "--device", args.device,
    ], commands)
    worker_payload = _json(worker_metrics_path)
    worker = worker_payload["metrics"]
    nested = torch.load(
        worker_candidates, map_location="cpu", weights_only=True
    )["nested_validity"]["metrics"]
    sentence_admission = root / "gates/sentence_admission.json"
    _admit(
        sentence_checkpoint, worker_payload["dataset_fingerprint"],
        "ORACLE_WAYPOINT", {**nested, **worker},
        worker["exact_symbolic_waypoint_success"] >= 0.5
        and worker["latent_symbolic_disagreement"] <= 0.25,
        sentence_admission,
    )
    commutation_dir = root / "models/commutation"
    _train(
        python, commands, features=train_features,
        counterfactual=counterfactual, output=commutation_dir,
        config="hierarchical_language_dynamic_commutation.yaml",
        steps=args.commutation_steps, learning_rate=1e-4,
        seed=args.seed, device=args.device,
        init=sentence_checkpoint, admission=sentence_admission,
    )
    commutation_checkpoint = commutation_dir / "model.pt"
    commutation_history = _json(commutation_dir / "metrics.json")["history"]
    commutation_error = float(commutation_history[-1]["commutation"])
    dynamic_worker_candidates = root / "gates/dynamic_worker_candidates.pt"
    dynamic_worker_metrics_path = root / "gates/dynamic_worker_metrics.json"
    _run([
        python, "scripts/build_sentence_waypoint_candidates.py",
        "--features", str(validation_features), "--examples",
        str(args.source_root / "splits/id_validation.jsonl"),
        "--checkpoint", str(commutation_checkpoint), "--output",
        str(dynamic_worker_candidates), "--dataset-split", "id_validation",
        "--k0", "64", "--population", "32", "--max-roots", "64",
        "--seed", str(args.seed), "--device", args.device,
        "--dtype", args.dtype,
    ], commands)
    _run([
        python, "scripts/evaluate_hierarchical_language_oracles.py",
        "--checkpoint", str(commutation_checkpoint), "--candidates",
        str(dynamic_worker_candidates), "--output",
        str(dynamic_worker_metrics_path), "--mode", "sentence_worker",
        "--metric", "mahalanobis", "--device", args.device,
    ], commands)
    dynamic_worker_payload = _json(dynamic_worker_metrics_path)
    dynamic_worker = dynamic_worker_payload["metrics"]
    dynamic_admission = root / "gates/commutation_admission.json"
    _admit(
        commutation_checkpoint, worker_payload["dataset_fingerprint"],
        "DYNAMIC_COMMUTATION", {
            "commutation_error": commutation_error,
            "worker_success": dynamic_worker[
                "predicted_symbolic_waypoint_success"
            ],
        },
        math.isfinite(commutation_error)
        and dynamic_worker["predicted_symbolic_waypoint_success"] > 0,
        dynamic_admission,
    )
    macro_dir = root / "models/macro"
    _train(
        python, commands, features=train_features,
        counterfactual=counterfactual, output=macro_dir,
        config="hierarchical_language_macro_action.yaml",
        steps=args.macro_steps, learning_rate=3e-4,
        seed=args.seed, device=args.device,
        init=commutation_checkpoint, admission=dynamic_admission,
    )
    macro_checkpoint = macro_dir / "model.pt"
    validation_rollouts = root / "gates/macro_validation_rollouts.pt"
    _run([
        python, "scripts/build_hierarchical_oracle_rollouts.py",
        "--features", str(validation_features), "--checkpoint",
        str(macro_checkpoint), "--output", str(validation_rollouts),
        "--max-roots", "128", "--first-actions", "16",
        "--continuation-samples", "1", "--max-prefix", "1",
        "--seed", str(args.seed), "--device", args.device,
    ], commands)
    # Oracle manager validity: actual worker execution, exact re-encoding,
    # and grounded CEM updates at K1=1 versus K1=4.
    oracle_gate = []
    for k1 in (1, 4):
        output = root / f"gates/oracle_mpc_k1_{k1}.json"
        _run([
            python, "scripts/evaluate_hierarchical_language_mpc.py",
            "--features", str(validation_features), "--examples",
            str(args.source_root / "splits/id_validation.jsonl"),
            "--checkpoint", str(macro_checkpoint), "--output", str(output),
            "--dataset-split", "id_validation", "--mode", "oracle",
            "--metric", "mahalanobis", "--k0", "64", "--k1", str(k1),
            "--max-examples", str(args.eval_examples), "--seed", str(args.seed),
            "--device", args.device, "--dtype", args.dtype,
        ], commands)
        oracle_gate.append(_json(output))
    oracle_gain = oracle_gate[1]["accuracy"] - oracle_gate[0]["accuracy"]
    optimizer_regret = oracle_gate[1]["mean_optimizer_curse_regret"]
    rollout_payload = torch.load(
        validation_rollouts, map_location="cpu", weights_only=True
    )
    macro_prior_nll = -float(
        rollout_payload["first_action_log_probability"].mean()
    )
    oracle_admission = root / "gates/oracle_high_admission.json"
    _admit(
        macro_checkpoint, oracle_gate[1]["dataset_fingerprint"],
        "ORACLE_HIGH_LEVEL", {
            "oracle_success_gain": oracle_gain,
            "optimizer_curse_regret": optimizer_regret,
            "macro_prior_nll": macro_prior_nll,
            "worker_executability": 1 - oracle_gate[1][
                "generation_failure_rate"
            ],
        },
        oracle_gate[1]["accuracy"] > 0 and oracle_gain >= 0
        and math.isfinite(optimizer_regret),
        oracle_admission,
    )
    # Only after oracle high-level planning is admitted do we construct the
    # training terminal-search artifact used by the value student.
    rollouts = root / "value/oracle_rollouts.pt"
    _run([
        python, "scripts/build_hierarchical_oracle_rollouts.py",
        "--features", str(train_features), "--checkpoint",
        str(macro_checkpoint), "--output", str(rollouts),
        "--max-roots", str(args.value_roots), "--first-actions", "16",
        "--continuation-samples", "4", "--max-prefix", "8",
        "--seed", str(args.seed), "--device", args.device,
    ], commands)
    value_checkpoints = {}
    for metric in ("euclidean", "mahalanobis"):
        for teacher, max_prefix in (("raw", 1), ("quasimetric", 8)):
            name = f"{metric}-{teacher}"
            replay = root / f"value/{name}.pt"
            _run([
                python, "scripts/build_hierarchical_value_teacher.py",
                "--checkpoint", str(macro_checkpoint), "--oracle-rollouts",
                str(rollouts), "--output", str(replay), "--metric", metric,
                "--max-prefix", str(max_prefix), "--device", args.device,
                "--step-cost", "0.0" if teacher == "raw" else "0.01",
                "--prior-weight", "0.0" if teacher == "raw" else "0.1",
            ], commands)
            directory = root / f"models/value-{name}"
            _train(
                python, commands, features=train_features,
                counterfactual=None, output=directory,
                config="hierarchical_language_value_distillation.yaml",
                steps=args.value_steps, learning_rate=3e-4,
                seed=args.seed, device=args.device,
                init=macro_checkpoint, admission=oracle_admission,
                value_replay=replay,
            )
            value_checkpoints[name] = directory / "model.pt"
    methods = {
        "oracle-euclidean": (macro_checkpoint, "oracle", "euclidean"),
        "oracle-mahalanobis": (macro_checkpoint, "oracle", "mahalanobis"),
        **{
            f"value-{name}": (checkpoint, "value", name.split("-")[0])
            for name, checkpoint in value_checkpoints.items()
        },
    }
    evaluations = []
    for method, (checkpoint, mode, metric) in methods.items():
        for split in SPLITS:
            for k0, k1 in DEPTH_PAIRS:
                output = root / "evaluations" / (
                    f"{method}-{split}-k0_{k0}-k1_{k1}.json"
                )
                _run([
                    python, "scripts/evaluate_hierarchical_language_mpc.py",
                    "--features", str(args.source_root / f"features/{split}.pt"),
                    "--examples", str(args.source_root / f"splits/{split}.jsonl"),
                    "--checkpoint", str(checkpoint), "--output", str(output),
                    "--dataset-split", split, "--mode", mode,
                    "--method-label", method,
                    "--metric", metric, "--k0", str(k0), "--k1", str(k1),
                    "--max-examples", str(args.eval_examples),
                    "--seed", str(args.seed), "--device", args.device,
                    "--dtype", args.dtype,
                ], commands)
                evaluations.append(output)
    _run([
        python, "scripts/plot_hierarchical_mpc_effort.py",
        *sum((["--evaluation", str(path)] for path in evaluations), []),
        "--output-prefix", str(root / "figures/mpc_effort_accuracy"),
    ], commands)
    checkpoint_evaluations = []
    for name, final_checkpoint in value_checkpoints.items():
        metric = name.split("-")[0]
        checkpoint_dir = final_checkpoint.parent / "checkpoints"
        for checkpoint in sorted(checkpoint_dir.glob("step_*.pt")):
            for split in SPLITS[1:]:
                output = root / "checkpoint_evaluations" / (
                    f"value-{name}-{checkpoint.stem}-{split}.json"
                )
                _run([
                    python, "scripts/evaluate_hierarchical_language_mpc.py",
                    "--features", str(args.source_root / f"features/{split}.pt"),
                    "--examples", str(args.source_root / f"splits/{split}.jsonl"),
                    "--checkpoint", str(checkpoint), "--output", str(output),
                    "--dataset-split", split, "--method-label", f"value-{name}",
                    "--mode", "value", "--metric", metric,
                    "--k0", "32", "--k1", "2",
                    "--max-examples", str(args.eval_examples),
                    "--seed", str(args.seed), "--device", args.device,
                    "--dtype", args.dtype,
                ], commands)
                checkpoint_evaluations.append(output)
    _run([
        python, "scripts/plot_hierarchical_mpc_checkpoints.py",
        *sum((
            ["--evaluation", str(path)]
            for path in checkpoint_evaluations
        ), []),
        "--output-prefix", str(root / "figures/mpc_ood_over_training"),
    ], commands)
    _write_json(root / "workflow.json", {
        "commands": commands,
        "methods": {name: str(checkpoint) for name, checkpoint in {
            **value_checkpoints, "macro": macro_checkpoint,
        }.items()},
        "depth_pairs": DEPTH_PAIRS,
        "splits": SPLITS,
        "seed": args.seed,
    })
    _write_json(root / "outcome.json", {
        "status": "complete", "seed": args.seed,
        "full_hierarchy_evaluated": True,
    })


if __name__ == "__main__":
    try:
        main()
    except ValidityGateFailure as error:
        arguments = parse_args()
        output_root = arguments.output_root or Path(os.environ.get("RUN_DIR", ""))
        if not str(output_root):
            raise
        _write_json(output_root / "outcome.json", {
            "status": "validity_gate_stop",
            "reason": str(error),
            "seed": arguments.seed,
            "full_hierarchy_evaluated": False,
        })
        print(json.dumps({
            "status": "validity_gate_stop", "reason": str(error)
        }), flush=True)
