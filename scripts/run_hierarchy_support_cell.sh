#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RUN_NAME CHECKPOINT METHOD ENERGY"
  exit 2
fi

run_name=$1
checkpoint=$2
method=$3
energy=$4
log_dir=research/intent_phrase/logs
mkdir -p "$log_dir"
rm -f "$log_dir/${run_name}.done" "$log_dir/${run_name}.failed"
trap 'touch "$log_dir/${run_name}.failed"' ERR

samples=1024
if [[ "$method" == "cem" ]]; then
  samples=1200
fi
.venv/bin/python scripts/audit_hierarchy_support.py \
  --ckpt "$checkpoint" \
  --out "runs/${run_name}/support.json" \
  --device cuda:0 \
  --method "$method" \
  --energy "$energy" \
  --anchors 100 \
  --samples "$samples" \
  --iters 20 \
  --elites 10 \
  --high-horizon 2 \
  --max-expand 256 \
  --density-weight 0.1

rm -f "$log_dir/${run_name}.failed"
touch "$log_dir/${run_name}.done"
