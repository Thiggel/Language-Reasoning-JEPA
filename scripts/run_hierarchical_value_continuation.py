#!/usr/bin/env python3
"""Continue an admitted nested sentence JEPA through macro and value stages.

This runner intentionally reuses a completed jointly trained sentence
checkpoint.  It performs the remaining causal stages with worker-grounded,
exactly re-encoded oracle rollouts and emits both an oracle-manager checkpoint
and a deployable no-terminal value checkpoint.
"""

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

from textjepa.training.hierarchical_language import ResearchStage


CONFIGS = {
    "euclidean-vicreg": (
        "hierarchical_language_dynamic_commutation_euclidean.yaml",
        "hierarchical_language_macro_action_euclidean.yaml",
        "hierarchical_language_value_euclidean.yaml",
        "euclidean",
    ),
    "mahalanobis-vicreg": (
        "hierarchical_language_dynamic_commutation_mahalanobis.yaml",
        "hierarchical_language_macro_action_mahalanobis.yaml",
        "hierarchical_language_value_mahalanobis.yaml",
        "mahalanobis",
    ),
    "euclidean-sigreg": (
        "hierarchical_language_dynamic_commutation_sigreg.yaml",
        "hierarchical_language_macro_action_sigreg.yaml",
        "hierarchical_language_value_sigreg.yaml",
        "euclidean",
    ),
}


class ValidityGateFailure(RuntimeError):
    """A measured scientific stop rather than an infrastructure failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sentence-checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--commutation-steps", type=int, default=50000)
    parser.add_argument("--macro-steps", type=int, default=50000)
    parser.add_argument("--value-steps", type=int, default=50000)
    parser.add_argument("--value-roots", type=int, default=256)
    parser.add_argument("--validation-roots", type=int, default=64)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--worker-population", type=int, default=16)
    parser.add_argument("--also-raw-value", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def _run(command: list[str], log: list[dict]) -> None:
    started = perf_counter()
    subprocess.run(command, check=True)
    log.append({"command": command, "wall_seconds": perf_counter() - started})


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _admit(
    checkpoint: Path,
    dataset_fingerprint: str,
    stage: str,
    metrics: dict[str, float],
    passed: bool,
    output: Path,
) -> None:
    finite = all(math.isfinite(float(value)) for value in metrics.values())
    record = {
        "passed": bool(passed and finite),
        "admitted_stage": stage,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "dataset_fingerprint": dataset_fingerprint,
        "evaluation_dataset_fingerprint": dataset_fingerprint,
        "admitted_checkpoint_sha256": _sha(checkpoint),
    }
    _write(output, record)
    if not record["passed"]:
        raise ValidityGateFailure(f"{stage} gate failed: {metrics}")


def _train(
    python: str,
    commands: list[dict],
    *,
    features: Path,
    output: Path,
    config: str,
    steps: int,
    learning_rate: float,
    seed: int,
    device: str,
    init: Path,
    admission: Path,
    counterfactual: Path | None = None,
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
        "--device", device, "--init-checkpoint", str(init),
        "--admission", str(admission),
    ]
    if counterfactual is not None:
        command.extend(["--counterfactual-features", str(counterfactual)])
    if value_replay is not None:
        command.extend(["--value-replay", str(value_replay)])
    _run(command, commands)


def _worker_gate(
    python: str,
    commands: list[dict],
    *,
    source: Path,
    checkpoint: Path,
    output_root: Path,
    metric: str,
    label: str,
    maximum_roots: int,
    population: int,
    seed: int,
    device: str,
    dtype: str,
) -> tuple[dict, dict, str]:
    candidates = output_root / "candidates.pt"
    metrics_path = output_root / "metrics.json"
    _run([
        python, "scripts/build_sentence_waypoint_candidates.py",
        "--features", str(source / "features/id_validation.pt"),
        "--examples", str(source / "splits/id_validation.jsonl"),
        "--checkpoint", str(checkpoint), "--output", str(candidates),
        "--dataset-split", "id_validation", "--k0", "64",
        "--population", str(population), "--max-roots", str(maximum_roots),
        "--seed", str(seed), "--device", device, "--dtype", dtype,
    ], commands)
    payload = torch.load(candidates, map_location="cpu", weights_only=True)
    nested = payload["nested_validity"]
    if nested.get("passed") is not True:
        raise ValidityGateFailure(
            f"nested representation gate failed: {nested.get('metrics')}"
        )
    _run([
        python, "scripts/evaluate_hierarchical_language_oracles.py",
        "--checkpoint", str(checkpoint), "--candidates", str(candidates),
        "--output", str(metrics_path), "--mode", "sentence_worker",
        "--metric", metric, "--device", device,
    ], commands)
    result = _json(metrics_path)
    return nested["metrics"], result["metrics"], result["dataset_fingerprint"]


def _mpc(
    python: str,
    commands: list[dict],
    *,
    source: Path,
    checkpoint: Path,
    output: Path,
    mode: str,
    metric: str,
    label: str,
    k1: int,
    examples: int,
    seed: int,
    device: str,
    dtype: str,
) -> dict:
    _run([
        python, "scripts/evaluate_hierarchical_language_mpc.py",
        "--features", str(source / "features/id_validation.pt"),
        "--examples", str(source / "splits/id_validation.jsonl"),
        "--checkpoint", str(checkpoint), "--output", str(output),
        "--dataset-split", "id_validation", "--method-label", label,
        "--mode", mode, "--metric", metric, "--k0", "64",
        "--k1", str(k1), "--max-examples", str(examples),
        "--seed", str(seed), "--device", device, "--dtype", dtype,
    ], commands)
    return _json(output)


def main() -> None:
    args = parse_args()
    sizes = (
        args.commutation_steps, args.macro_steps, args.value_steps,
        args.value_roots, args.validation_roots, args.eval_examples,
        args.worker_population,
    )
    if min(sizes) < 1:
        raise ValueError("continuation sizes must be positive")
    run_dir = os.environ.get("RUN_DIR")
    root = args.output_root or (Path(run_dir) if run_dir else None)
    if root is None:
        raise ValueError("--output-root or RUN_DIR is required")
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_header = torch.load(
        args.sentence_checkpoint, map_location="cpu", weights_only=True
    )
    if checkpoint_header.get("stage") != ResearchStage.SENTENCE_JEPA.name:
        raise ValueError("continuation requires a SENTENCE_JEPA checkpoint")
    dynamic_config, macro_config, value_config, metric = CONFIGS[args.variant]
    python = sys.executable
    commands: list[dict] = []
    train_features = args.source_root / "features/train.pt"
    counterfactual = args.source_root / "features/train_counterfactual.pt"
    for path in (
        train_features, counterfactual,
        args.source_root / "features/id_validation.pt",
        args.source_root / "splits/train.jsonl",
        args.source_root / "splits/id_validation.jsonl",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    nested, worker, fingerprint = _worker_gate(
        python, commands, source=args.source_root,
        checkpoint=args.sentence_checkpoint,
        output_root=root / "gates/sentence-worker", metric=metric,
        label=args.method_label, maximum_roots=args.validation_roots,
        population=args.worker_population, seed=args.seed,
        device=args.device, dtype=args.dtype,
    )
    sentence_admission = root / "gates/sentence_admission.json"
    sentence_metrics = {**nested, **worker}
    _admit(
        args.sentence_checkpoint, fingerprint, "ORACLE_WAYPOINT",
        sentence_metrics,
        nested["heldout_sentence_dynamics_gain"] > 0
        and nested["heldout_sentence_dynamics"] < (
            0.9 * nested["heldout_sentence_identity"]
        )
        and worker["exact_symbolic_waypoint_success"] >= 0.5
        and worker["latent_symbolic_disagreement"] <= 0.25,
        sentence_admission,
    )

    commutation_dir = root / "models/commutation"
    _train(
        python, commands, features=train_features,
        counterfactual=counterfactual, output=commutation_dir,
        config=dynamic_config, steps=args.commutation_steps,
        learning_rate=1e-4, seed=args.seed, device=args.device,
        init=args.sentence_checkpoint, admission=sentence_admission,
    )
    commutation_checkpoint = commutation_dir / "model.pt"
    dynamic_nested, dynamic_worker, dynamic_fingerprint = _worker_gate(
        python, commands, source=args.source_root,
        checkpoint=commutation_checkpoint,
        output_root=root / "gates/commutation-worker", metric=metric,
        label=args.method_label, maximum_roots=args.validation_roots,
        population=args.worker_population, seed=args.seed,
        device=args.device, dtype=args.dtype,
    )
    baseline_error = nested["heldout_commutation_error"]
    dynamic_error = dynamic_nested["heldout_commutation_error"]
    commutation_admission = root / "gates/commutation_admission.json"
    _admit(
        commutation_checkpoint, dynamic_fingerprint, "DYNAMIC_COMMUTATION",
        {
            "commutation_error": dynamic_error,
            "commutation_baseline": baseline_error,
            "commutation_relative_gain": (
                (baseline_error - dynamic_error) / max(baseline_error, 1e-8)
            ),
            "worker_success": dynamic_worker[
                "predicted_symbolic_waypoint_success"
            ],
        },
        dynamic_error <= 0.95 * baseline_error
        and dynamic_worker["predicted_symbolic_waypoint_success"] >= (
            0.9 * worker["predicted_symbolic_waypoint_success"]
        ),
        commutation_admission,
    )

    macro_dir = root / "models/macro"
    _train(
        python, commands, features=train_features,
        counterfactual=counterfactual, output=macro_dir,
        config=macro_config, steps=args.macro_steps,
        learning_rate=3e-4, seed=args.seed, device=args.device,
        init=commutation_checkpoint, admission=commutation_admission,
    )
    macro_checkpoint = macro_dir / "model.pt"
    validation_rollouts = root / "value/validation_rollouts.pt"
    _run([
        python, "scripts/build_hierarchical_oracle_rollouts.py",
        "--features", str(args.source_root / "features/id_validation.pt"),
        "--examples", str(args.source_root / "splits/id_validation.jsonl"),
        "--checkpoint", str(macro_checkpoint), "--output",
        str(validation_rollouts), "--max-roots", str(args.validation_roots),
        "--first-actions", "4", "--continuation-samples", "1",
        "--max-prefix", "8", "--worker-population", "8", "--k0", "64",
        "--seed", str(args.seed), "--device", args.device,
        "--dtype", args.dtype,
    ], commands)
    oracle_one = _mpc(
        python, commands, source=args.source_root,
        checkpoint=macro_checkpoint, output=root / "gates/oracle-k1.json",
        mode="oracle", metric=metric, label=args.method_label,
        k1=1, examples=args.eval_examples, seed=args.seed,
        device=args.device, dtype=args.dtype,
    )
    oracle_four = _mpc(
        python, commands, source=args.source_root,
        checkpoint=macro_checkpoint, output=root / "gates/oracle-k4.json",
        mode="oracle", metric=metric, label=args.method_label,
        k1=4, examples=args.eval_examples, seed=args.seed,
        device=args.device, dtype=args.dtype,
    )
    validation_payload = torch.load(
        validation_rollouts, map_location="cpu", weights_only=True
    )
    macro_nll = -float(validation_payload["first_action_log_probability"][
        validation_payload["action_mask"]
    ].mean())
    executability = float(
        validation_payload["rollout_symbolically_verified"].float().mean()
    )
    oracle_admission = root / "gates/oracle_high_admission.json"
    _admit(
        macro_checkpoint, oracle_four["dataset_fingerprint"],
        "ORACLE_HIGH_LEVEL", {
            "oracle_success_gain": (
                oracle_four["accuracy"] - oracle_one["accuracy"]
            ),
            "optimizer_curse_regret": oracle_four[
                "mean_optimizer_curse_regret"
            ],
            "macro_prior_nll": macro_nll,
            "worker_executability": executability,
        },
        oracle_four["accuracy"] > 0
        and oracle_four["accuracy"] >= oracle_one["accuracy"]
        and math.isfinite(macro_nll)
        and executability >= 0.10,
        oracle_admission,
    )

    train_rollouts = root / "value/train_rollouts.pt"
    _run([
        python, "scripts/build_hierarchical_oracle_rollouts.py",
        "--features", str(train_features), "--examples",
        str(args.source_root / "splits/train.jsonl"), "--checkpoint",
        str(macro_checkpoint), "--output", str(train_rollouts),
        "--max-roots", str(args.value_roots), "--first-actions", "4",
        "--continuation-samples", "1", "--max-prefix", "8",
        "--worker-population", "8", "--k0", "64",
        "--seed", str(args.seed), "--device", args.device,
        "--dtype", args.dtype,
    ], commands)
    teachers = {"quasimetric": 8}
    if args.also_raw_value:
        teachers["raw"] = 1
    value_results = {}
    for teacher, max_prefix in teachers.items():
        replay = root / f"value/{teacher}.pt"
        heldout_replay = root / f"value/heldout-{teacher}.pt"
        teacher_args = [
            "--metric", metric, "--max-prefix", str(max_prefix),
            "--device", args.device,
            "--step-cost", "0.0" if teacher == "raw" else "0.01",
            "--prior-weight", "0.0" if teacher == "raw" else "0.1",
        ]
        for rollouts, output in (
            (train_rollouts, replay),
            (validation_rollouts, heldout_replay),
        ):
            _run([
                python, "scripts/build_hierarchical_value_teacher.py",
                "--checkpoint", str(macro_checkpoint),
                "--oracle-rollouts", str(rollouts),
                "--output", str(output), *teacher_args,
            ], commands)
        value_dir = root / f"models/value-{teacher}"
        _train(
            python, commands, features=train_features, output=value_dir,
            config=value_config, steps=args.value_steps,
            learning_rate=3e-4, seed=args.seed, device=args.device,
            init=macro_checkpoint, admission=oracle_admission,
            value_replay=replay,
        )
        value_checkpoint = value_dir / "model.pt"
        value_metrics_path = root / f"gates/value-{teacher}.json"
        _run([
            python, "scripts/evaluate_hierarchical_value_replay.py",
            "--checkpoint", str(value_checkpoint), "--value-replay",
            str(heldout_replay), "--output", str(value_metrics_path),
            "--device", args.device,
        ], commands)
        ranking = _json(value_metrics_path)["metrics"]
        if (
            ranking["pairwise_ranking_accuracy"] <= 0.5
            or ranking["spearman"] <= 0
            or not math.isfinite(ranking["top_one_regret"])
        ):
            raise ValidityGateFailure(
                f"held-out {teacher} value ranking failed: {ranking}"
            )
        value_mpc = {}
        for k1 in (1, 2, 4, 8):
            value_mpc[k1] = _mpc(
                python, commands, source=args.source_root,
                checkpoint=value_checkpoint,
                output=root / f"gates/value-{teacher}-k{k1}.json",
                mode="value", metric=metric,
                label=f"{args.method_label}-{teacher}", k1=k1,
                examples=args.eval_examples, seed=args.seed,
                device=args.device, dtype=args.dtype,
            )["accuracy"]
        value_results[teacher] = {
            "checkpoint": str(value_checkpoint),
            "checkpoint_sha256": _sha(value_checkpoint),
            "ranking": ranking,
            "validation_accuracy_by_k1": value_mpc,
        }

    _write(root / "workflow.json", {
        "commands": commands,
        "variant": args.variant,
        "method_label": args.method_label,
        "source_sentence_checkpoint": str(args.sentence_checkpoint),
        "source_sentence_checkpoint_sha256": _sha(args.sentence_checkpoint),
        "macro_checkpoint": str(macro_checkpoint),
        "macro_checkpoint_sha256": _sha(macro_checkpoint),
        "oracle_validation_accuracy": {
            "k1_1": oracle_one["accuracy"],
            "k1_4": oracle_four["accuracy"],
        },
        "values": value_results,
    })
    _write(root / "outcome.json", {
        "status": "complete", "variant": args.variant,
        "method_label": args.method_label,
        "macro_checkpoint": str(macro_checkpoint),
        "value_checkpoints": {
            name: payload["checkpoint"]
            for name, payload in value_results.items()
        },
    })


if __name__ == "__main__":
    try:
        main()
    except ValidityGateFailure as error:
        arguments = parse_args()
        run_dir = os.environ.get("RUN_DIR")
        output_root = arguments.output_root or (
            Path(run_dir) if run_dir else None
        )
        if output_root is None:
            raise
        _write(output_root / "outcome.json", {
            "status": "validity_gate_stop",
            "reason": str(error),
            "variant": arguments.variant,
            "method_label": arguments.method_label,
        })
        print(json.dumps({
            "status": "validity_gate_stop", "reason": str(error)
        }), flush=True)
