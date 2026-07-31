#!/usr/bin/env python3
"""Run the staged strict-nested iGSM hierarchy and its 2x2 ablation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
DEPTH_PAIRS = tuple(itertools.product((8, 16, 32, 64), (1, 2, 4, 8)))


class ValidityGateFailure(RuntimeError):
    """A scientifically valid early stop, distinct from a runtime failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--token-steps", type=int, default=50000)
    parser.add_argument("--sentence-steps", type=int, default=50000)
    parser.add_argument("--commutation-steps", type=int, default=50000)
    parser.add_argument("--macro-steps", type=int, default=50000)
    parser.add_argument("--value-steps", type=int, default=50000)
    parser.add_argument("--reanalysis-steps", type=int, default=50000)
    parser.add_argument("--value-roots", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=64)
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
    planner_replay: Path | None = None,
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
    if planner_replay is not None:
        command.extend(["--planner-replay", str(planner_replay)])
    _run(command, commands)


def main() -> None:
    args = parse_args()
    if min(
        args.token_steps, args.sentence_steps, args.commutation_steps,
        args.macro_steps, args.value_steps, args.reanalysis_steps,
        args.value_roots,
        args.eval_examples,
    ) < 1:
        raise ValueError("experiment scales must be positive")
    run_dir = os.environ.get("RUN_DIR")
    root = args.output_root or (Path(run_dir) if run_dir else None)
    if root is None:
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
    proposal_path = root / "gates/proposal_coverage.json"
    _run([
        python, "scripts/evaluate_hierarchical_proposal_coverage.py",
        "--features", str(validation_features), "--examples",
        str(args.source_root / "splits/id_validation.jsonl"),
        "--output", str(proposal_path), "--population", "32",
        "--k0", "64", "--max-roots", "128", "--seed", str(args.seed),
        "--device", args.device, "--dtype", args.dtype,
    ], commands)
    proposal = _json(proposal_path)["metrics"]
    if proposal["oracle@32"] <= proposal["greedy"] or (
        proposal["oracle@32"] < 0.10
    ):
        raise ValidityGateFailure(
            f"frozen-LM proposal coverage is insufficient: {proposal}"
        )
    token_dir = root / "models/token"
    _train(
        python, commands, features=train_features,
        counterfactual=counterfactual, output=token_dir,
        config="hierarchical_language_token_multistep.yaml",
        steps=args.token_steps, learning_rate=1e-3,
        seed=args.seed, device=args.device,
    )
    token_checkpoint = token_dir / "model.pt"
    token_health_path = root / "gates/token_health.json"
    _run([
        python, "scripts/evaluate_hierarchical_token_health.py",
        "--features", str(validation_features), "--checkpoint",
        str(token_checkpoint), "--output", str(token_health_path),
        "--max-examples", "256", "--device", args.device,
    ], commands)
    token_health = _json(token_health_path)["metrics"]
    if any(
        token_health[name] <= 1.05
        for name in ("identity_to_full", "action_only_to_full", "cache_only_to_full")
    ):
        raise ValidityGateFailure(
            f"token predictor fails causal-state shortcut gate: {token_health}"
        )
    flat = {}
    flat_payload = None
    for horizon in (4, 8, 16):
        flat_candidates = root / f"gates/flat_candidates_k{horizon}.pt"
        flat_metrics_path = root / f"gates/flat_metrics_k{horizon}.json"
        _run([
            python, "scripts/build_flat_token_oracle_candidates.py",
            "--features", str(validation_features),
            "--checkpoint", str(token_checkpoint),
            "--output", str(flat_candidates), "--dataset-split", "id_validation",
            "--method-label", "full-token", "--horizon", str(horizon),
            "--population", "64", "--max-roots", "128",
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
        for name, value in flat_payload["metrics"].items():
            flat[f"k{horizon}_{name}"] = value
        gain = flat_payload["metrics"]["exact_oracle_gain"]
        flat[f"k{horizon}_relative_model_gap"] = (
            flat_payload["metrics"]["model_oracle_gap"] / max(gain, 1e-8)
        )
    assert flat_payload is not None
    flat["exact_oracle_gain"] = flat["k8_exact_oracle_gain"]
    flat["model_oracle_gap"] = flat["k8_model_oracle_gap"]
    flat_admission = root / "gates/flat_admission.json"
    _admit(
        token_checkpoint, flat_payload["dataset_fingerprint"],
        "FLAT_ORACLE_TOKEN", {**flat, **token_health},
        all(flat[f"k{k}_exact_oracle_gain"] > 0 for k in (4, 8, 16))
        and all(
            flat[f"k{k}_relative_model_gap"] <= 1.0 for k in (4, 8, 16)
        )
        and flat["k8_predicted_next_token_accuracy"] > 0,
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
    geometry_artifact = root / "diagnostics/sentence_geometry.pt"
    geometry_plots = root / "figures/sentence_geometry"
    _run([
        python, "scripts/build_hierarchical_sentence_geometry.py",
        "--features", str(validation_features), "--examples",
        str(args.source_root / "splits/id_validation.jsonl"),
        "--checkpoint", str(sentence_checkpoint), "--output",
        str(geometry_artifact), "--plot-output-dir", str(geometry_plots),
        "--max-igsm-groups", "512", "--device", args.device,
        "--dtype", args.dtype,
    ], commands)
    _run([
        python, "scripts/plot_hierarchical_sentence_geometry.py",
        "--artifact", str(geometry_artifact), "--output-dir",
        str(geometry_plots), "--method", "umap", "--seed", str(args.seed),
    ], commands)
    worker_by_k0 = {}
    worker_payload = None
    nested = None
    for k0 in (8, 16, 32, 64):
        worker_candidates = root / f"gates/sentence_worker_k0_{k0}.pt"
        worker_metrics_path = root / f"gates/sentence_worker_k0_{k0}.json"
        _run([
            python, "scripts/build_sentence_waypoint_candidates.py",
            "--features", str(validation_features), "--examples",
            str(args.source_root / "splits/id_validation.jsonl"),
            "--checkpoint", str(sentence_checkpoint), "--output",
            str(worker_candidates), "--dataset-split", "id_validation",
            "--k0", str(k0), "--population", "32", "--max-roots", "128",
            "--seed", str(args.seed), "--device", args.device,
            "--dtype", args.dtype,
        ], commands)
        candidate_payload = torch.load(
            worker_candidates, map_location="cpu", weights_only=True
        )
        if candidate_payload["nested_validity"]["passed"] is not True:
            raise ValidityGateFailure(
                f"nested sentence gate failed at K0={k0}: "
                f"{candidate_payload['nested_validity']['metrics']}"
            )
        _run([
            python, "scripts/evaluate_hierarchical_language_oracles.py",
            "--checkpoint", str(sentence_checkpoint), "--candidates",
            str(worker_candidates), "--output", str(worker_metrics_path),
            "--mode", "sentence_worker", "--metric", "mahalanobis",
            "--device", args.device,
        ], commands)
        worker_payload = _json(worker_metrics_path)
        worker_by_k0[k0] = worker_payload["metrics"]
        nested = candidate_payload["nested_validity"]["metrics"]
    assert worker_payload is not None and nested is not None
    worker = worker_by_k0[64]
    sentence_admission = root / "gates/sentence_admission.json"
    _admit(
        sentence_checkpoint, worker_payload["dataset_fingerprint"],
        "ORACLE_WAYPOINT", {
            **nested,
            **worker,
            **{f"k0_{k}_{name}": value for k, values in worker_by_k0.items()
               for name, value in values.items()},
        },
        nested["heldout_sentence_dynamics_gain"] > 0
        and nested["heldout_sentence_dynamics"] < (
            0.9 * nested["heldout_sentence_identity"]
        )
        and worker["exact_symbolic_waypoint_success"] >= 0.5
        and worker["latent_symbolic_disagreement"] <= 0.25
        and worker_by_k0[64]["exact_symbolic_waypoint_success"] >= (
            worker_by_k0[8]["exact_symbolic_waypoint_success"]
        ),
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
    dynamic_nested = torch.load(
        dynamic_worker_candidates, map_location="cpu", weights_only=True
    )["nested_validity"]["metrics"]
    commutation_error = dynamic_nested["heldout_commutation_error"]
    commutation_baseline = nested["heldout_commutation_error"]
    dynamic_admission = root / "gates/commutation_admission.json"
    _admit(
        commutation_checkpoint, worker_payload["dataset_fingerprint"],
        "DYNAMIC_COMMUTATION", {
            "commutation_error": commutation_error,
            "commutation_baseline": commutation_baseline,
            "commutation_relative_gain": (
                (commutation_baseline - commutation_error)
                / max(commutation_baseline, 1e-8)
            ),
            "worker_success": dynamic_worker[
                "predicted_symbolic_waypoint_success"
            ],
        },
        math.isfinite(commutation_error)
        and commutation_error <= 0.95 * commutation_baseline
        and dynamic_worker["predicted_symbolic_waypoint_success"] >= (
            0.9 * worker["predicted_symbolic_waypoint_success"]
        ),
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
        "--features", str(validation_features), "--examples",
        str(args.source_root / "splits/id_validation.jsonl"), "--checkpoint",
        str(macro_checkpoint), "--output", str(validation_rollouts),
        "--max-roots", "64", "--first-actions", "4",
        "--continuation-samples", "1", "--max-prefix", "8",
        "--worker-population", "8", "--k0", "64",
        "--seed", str(args.seed), "--device", args.device,
        "--dtype", args.dtype,
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
        rollout_payload["first_action_log_probability"][
            rollout_payload["action_mask"]
        ].mean()
    )
    worker_executability = float(
        rollout_payload["rollout_symbolically_verified"][:, :, :, 0].float().mean()
    )
    mean_grounding_displacement = float(
        rollout_payload["requested_achieved_displacement"][:, :, :, 0].mean()
    )
    oracle_admission = root / "gates/oracle_high_admission.json"
    _admit(
        macro_checkpoint, oracle_gate[1]["dataset_fingerprint"],
        "ORACLE_HIGH_LEVEL", {
            "oracle_success_gain": oracle_gain,
            "optimizer_curse_regret": optimizer_regret,
            "macro_prior_nll": macro_prior_nll,
            "worker_executability": worker_executability,
            "mean_grounding_displacement": mean_grounding_displacement,
        },
        oracle_gate[1]["accuracy"] > 0 and oracle_gain >= 0
        and math.isfinite(optimizer_regret)
        and worker_executability >= 0.10
        and math.isfinite(macro_prior_nll)
        and math.isfinite(mean_grounding_displacement),
        oracle_admission,
    )
    # Only after oracle high-level planning is admitted do we construct the
    # training terminal-search artifact used by the value student.
    rollouts = root / "value/oracle_rollouts.pt"
    _run([
        python, "scripts/build_hierarchical_oracle_rollouts.py",
        "--features", str(train_features), "--examples",
        str(args.source_root / "splits/train.jsonl"), "--checkpoint",
        str(macro_checkpoint), "--output", str(rollouts),
        "--max-roots", str(args.value_roots), "--first-actions", "4",
        "--continuation-samples", "1", "--max-prefix", "8",
        "--worker-population", "8", "--k0", "64",
        "--seed", str(args.seed), "--device", args.device,
        "--dtype", args.dtype,
    ], commands)
    value_checkpoints = {}
    value_gate_metrics = {}
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
            value_checkpoint = directory / "model.pt"
            value_checkpoints[name] = value_checkpoint
            heldout_replay = root / f"value/heldout-{name}.pt"
            _run([
                python, "scripts/build_hierarchical_value_teacher.py",
                "--checkpoint", str(macro_checkpoint), "--oracle-rollouts",
                str(validation_rollouts), "--output", str(heldout_replay),
                "--metric", metric, "--max-prefix", str(max_prefix),
                "--device", args.device,
                "--step-cost", "0.0" if teacher == "raw" else "0.01",
                "--prior-weight", "0.0" if teacher == "raw" else "0.1",
            ], commands)
            heldout_metrics_path = root / f"gates/value-{name}.json"
            _run([
                python, "scripts/evaluate_hierarchical_value_replay.py",
                "--checkpoint", str(value_checkpoint), "--value-replay",
                str(heldout_replay), "--output", str(heldout_metrics_path),
                "--device", args.device,
            ], commands)
            value_gate_metrics[name] = _json(heldout_metrics_path)["metrics"]
    failed_values = {
        name: metrics for name, metrics in value_gate_metrics.items()
        if metrics["pairwise_ranking_accuracy"] <= 0.5
        or metrics["spearman"] <= 0.0
        or not math.isfinite(metrics["top_one_regret"])
    }
    if failed_values:
        raise ValidityGateFailure(
            f"held-out value ranking gate failed: {failed_values}"
        )
    full_value_checkpoint = value_checkpoints["mahalanobis-quasimetric"]
    full_value_gate_path = root / "gates/full_value_mpc.json"
    _run([
        python, "scripts/evaluate_hierarchical_language_mpc.py",
        "--features", str(validation_features), "--examples",
        str(args.source_root / "splits/id_validation.jsonl"),
        "--checkpoint", str(full_value_checkpoint), "--output",
        str(full_value_gate_path), "--dataset-split", "id_validation",
        "--mode", "value", "--method-label", "full-value-gate",
        "--metric", "mahalanobis", "--k0", "32", "--k1", "2",
        "--max-examples", str(max(128, args.eval_examples)),
        "--seed", str(args.seed), "--device", args.device,
        "--dtype", args.dtype,
    ], commands)
    full_value_gate = _json(full_value_gate_path)
    full_admission = root / "gates/full_hierarchy_admission.json"
    predicted_realized_gap = sum(
        row["mean_worker_exact_gap"] or 0.0
        for row in full_value_gate["rows"]
    ) / len(full_value_gate["rows"])
    _admit(
        full_value_checkpoint, full_value_gate["dataset_fingerprint"],
        "FULL_HIERARCHY", {
            "no_terminal_success": full_value_gate["accuracy"],
            "predicted_realized_gap": predicted_realized_gap,
        },
        full_value_gate["accuracy"] > 0
        and math.isfinite(predicted_realized_gap),
        full_admission,
    )
    planner_replay = root / "reanalysis/planner_replay.pt"
    planner_collection = root / "reanalysis/train_mpc.json"
    _run([
        python, "scripts/evaluate_hierarchical_language_mpc.py",
        "--features", str(train_features), "--examples",
        str(args.source_root / "splits/train.jsonl"),
        "--checkpoint", str(full_value_checkpoint), "--output",
        str(planner_collection), "--replay-output", str(planner_replay),
        "--dataset-split", "train", "--mode", "value",
        "--method-label", "planner-reanalysis-collector",
        "--metric", "mahalanobis", "--k0", "32", "--k1", "2",
        "--max-examples", "128", "--seed", str(args.seed),
        "--device", args.device, "--dtype", args.dtype,
    ], commands)
    reanalysis_dir = root / "models/closed-loop-reanalysis"
    _train(
        python, commands, features=train_features, counterfactual=None,
        output=reanalysis_dir,
        config="hierarchical_language_closed_loop.yaml",
        steps=args.reanalysis_steps, learning_rate=1e-5,
        seed=args.seed, device=args.device, init=full_value_checkpoint,
        admission=full_admission,
        value_replay=root / "value/mahalanobis-quasimetric.pt",
        planner_replay=planner_replay,
    )
    reanalysis_checkpoint = reanalysis_dir / "model.pt"
    methods = {
        "oracle-euclidean": (macro_checkpoint, "oracle", "euclidean"),
        "oracle-mahalanobis": (macro_checkpoint, "oracle", "mahalanobis"),
        **{
            f"value-{name}": (checkpoint, "value", name.split("-")[0])
            for name, checkpoint in value_checkpoints.items()
        },
        "value-mahalanobis-quasimetric-closed-loop": (
            reanalysis_checkpoint, "value", "mahalanobis"
        ),
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
    control_checkpoint = value_checkpoints["mahalanobis-quasimetric"]
    controls = [
        ("greedy-qwen", "greedy", macro_checkpoint, [(k0, 1) for k0 in (8, 16, 32, 64)]),
        ("flat-token-value", "flat_value", control_checkpoint, [(k0, 1) for k0 in (8, 16, 32, 64)]),
        ("prior-only-manager", "prior_only", macro_checkpoint, list(DEPTH_PAIRS)),
    ]
    for method, mode, checkpoint, pairs in controls:
        for split in SPLITS:
            for k0, k1 in pairs:
                output = root / "evaluations" / (
                    f"{method}-{split}-k0_{k0}-k1_{k1}.json"
                )
                _run([
                    python, "scripts/evaluate_hierarchical_language_controls.py",
                    "--features", str(args.source_root / f"features/{split}.pt"),
                    "--examples", str(args.source_root / f"splits/{split}.jsonl"),
                    "--checkpoint", str(checkpoint), "--output", str(output),
                    "--dataset-split", split, "--mode", mode,
                    "--method-label", method, "--metric", "mahalanobis",
                    "--k0", str(k0), "--k1", str(k1),
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
        "statistical_scope": (
            "single-seed decision-grade mechanism run; Wilson intervals are "
            "reported per cell and multi-seed confirmation is still required"
        ),
        "value_gate_metrics": value_gate_metrics,
        "proposal_coverage": proposal,
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
        run_dir = os.environ.get("RUN_DIR")
        output_root = arguments.output_root or (
            Path(run_dir) if run_dir else None
        )
        if output_root is None:
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
