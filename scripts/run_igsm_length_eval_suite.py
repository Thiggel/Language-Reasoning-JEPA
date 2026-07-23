"""Run a resumable, operation-count-stratified iGSM evaluation suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


STRATA = (
    ("op_le_12", 6, 12),
    ("op_eq_12", 12, 12),
    ("op_eq_16", 16, 16),
    ("op_eq_17", 17, 17),
    ("op_eq_18", 18, 18),
)


def run(command, destination):
    if destination.exists():
        return json.loads(destination.read_text())
    subprocess.run(command, check=True)
    return json.loads(destination.read_text())


def generative(kind, checkpoint, root, examples, max_tokens):
    rows = {}
    for offset, (name, low, high) in enumerate(STRATA):
        destination = root / f"{kind}_{name}.json"
        command = [
            sys.executable, "scripts/eval_generative_lm_baseline.py",
            "--kind", kind, "--ckpt", checkpoint, "--examples", str(examples),
            "--max-tokens", str(max_tokens), "--width", "1",
            "--eval-seed", str(720001 + offset * 1009),
            "--steps-min", str(low), "--steps-max", str(high),
            "--n-vars-min", "10", "--n-vars-max", "20",
            "--output", str(destination),
        ]
        rows[name] = run(command, destination)
    return rows


def jepa(kind, checkpoint, root, examples, max_tokens):
    rows = {}
    settings = (
        [("prior_only", 0, "prior", "prior", 1.0)]
        if kind == "jepa-prior" else []
    ) + [
        (
            "value_plan_d4", 4, "value",
            "prior" if kind == "jepa-prior" else "all",
            1.0 if kind == "jepa-prior" else 0.0,
        )
    ]
    for setting, depth, score, proposals, prior_weight in settings:
        rows[setting] = {}
        for offset, (name, low, high) in enumerate(STRATA):
            tag = f"{kind}_{setting}_{name}"
            destination = root / (
                f"pooled_{tag}_mpc_{proposals}_{score}_d{depth}_w8"
                f"_pw{prior_weight:g}_lmfixed_budget.json"
            )
            command = [
                sys.executable, "scripts/eval_pooled_sentence_planning.py",
                "--ckpt", checkpoint, "--examples", str(examples),
                "--max-tokens", str(max_tokens), "--length-mode", "fixed_budget",
                "--depth", str(depth), "--width", "8", "--planner", "mpc",
                "--score", score, "--proposals", proposals,
                "--proposal-topk", "20",
                "--prior-score-weight", str(prior_weight),
                "--eval-seed", str(720001 + offset * 1009),
                "--steps-min", str(low), "--steps-max", str(high),
                "--n-vars-min", "10", "--n-vars-max", "20",
                "--output-tag", tag, "--output-dir", str(root),
            ]
            rows[setting][name] = run(command, destination)
    collapse = root / f"{kind}_collapse.json"
    run([
        sys.executable, "scripts/diagnose_pooled_state_collapse.py",
        "--ckpt", checkpoint, "--output", str(collapse),
    ], collapse)
    probes = root / f"{kind}_representation_probes.json"
    run([
        sys.executable, "scripts/probe_pooled_sentence_states.py",
        "--ckpt", checkpoint, "--output", str(probes),
        "--train-examples", "512", "--test-examples", "256",
    ], probes)
    rows["collapse_diagnostics"] = json.loads(collapse.read_text())
    rows["representation_probes"] = json.loads(probes.read_text())
    return rows


def parse_model(value):
    try:
        kind, checkpoint = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("model must be KIND=/absolute/checkpoint") from exc
    if kind not in {"token", "sentence", "jepa-prior", "jepa-no-prior"}:
        raise argparse.ArgumentTypeError(f"unknown model kind: {kind}")
    return kind, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--examples-lm", type=int, default=64)
    parser.add_argument("--examples-jepa", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    root = Path(args.output_dir or os.environ["RUN_DIR"]) / "length_eval"
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "suite_summary.json"
    summary = {
        "protocol": {
            "difficulty": "number of necessary solution operations",
            "training_support": "op in [6,12]",
            "strata": [list(row) for row in STRATA],
            "n_vars_range": [10, 20],
            "generation": "no symbolic feedback; fixed token budget",
            "pilot_note": (
                "This dependency-held pass is a directional pilot. Scale each "
                "stratum after runtime and confidence are known."
            ),
        },
        "models": {},
    }
    for kind, checkpoint in args.model:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(checkpoint)
        if kind in {"token", "sentence"}:
            result = generative(
                kind, checkpoint, root, args.examples_lm, args.max_tokens
            )
        else:
            result = jepa(
                kind, checkpoint, root, args.examples_jepa, args.max_tokens
            )
        summary["models"][kind] = result
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
