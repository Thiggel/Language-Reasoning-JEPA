#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
ckpt=${2:?checkpoint path}
mode=${3:?follow-up mode}
parent=${RUN_DIR:?RUN_DIR must be supplied by researchctl}

"$python_bin" scripts/audit_token_planner_interface.py \
  --ckpt "$ckpt" --device cuda:0 --examples 32 --positions 128 \
  --topk 20 --goal-horizons 1 4 16 0 \
  --out "$parent/planner_interface_audit.json"

case "$mode" in
  asym-low1-high2)
    mkdir -p "$parent/training"
    RUN_DIR="$parent/training" bash scripts/run_dropout_rollout_screen_cell.sh \
      "$python_bin" "$mode" 0 1 1.0 2 1.0 1.0
    ;;
  asym-low1-high4)
    mkdir -p "$parent/training"
    RUN_DIR="$parent/training" bash scripts/run_dropout_rollout_screen_cell.sh \
      "$python_bin" "$mode" 0 1 1.0 4 1.0 0.5
    ;;
  asym-low1-high8)
    mkdir -p "$parent/training"
    RUN_DIR="$parent/training" bash scripts/run_dropout_rollout_screen_cell.sh \
      "$python_bin" "$mode" 0 1 1.0 8 1.0 0.7
    ;;
  asym-low4-high1)
    mkdir -p "$parent/training"
    RUN_DIR="$parent/training" bash scripts/run_dropout_rollout_screen_cell.sh \
      "$python_bin" "$mode" 0 4 0.5 1 0.5 1.0
    ;;
  counterfactual-density)
    mkdir -p "$parent/training"
    RUN_DIR="$parent/training" bash scripts/run_weekend_counterfactual_density_matrix.sh \
      "$python_bin"
    ;;
  audit-only)
    ;;
  *)
    echo "unknown follow-up mode: $mode" >&2
    exit 2
    ;;
esac

"$python_bin" - "$parent" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
result = {"planner_interface_audit": json.loads((root / "planner_interface_audit.json").read_text())}
training = root / "training" / "metrics.json"
if training.exists():
    result["training"] = json.loads(training.read_text())
(root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
PY
