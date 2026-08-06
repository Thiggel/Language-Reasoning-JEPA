#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
ckpt=${2:?checkpoint path}
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}

"$python_bin" scripts/audit_token_planner_interface.py \
  --ckpt "$ckpt" --device cuda:0 --examples 32 --positions 128 \
  --topk 20 --goal-horizons 1 4 16 0 \
  --out "$run_dir/planner_interface_audit.json"
cp "$run_dir/planner_interface_audit.json" "$run_dir/metrics.json"
