#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 RUN_NAME CHECKPOINT TAG [HYDRA_OVERRIDES...]"
  exit 2
fi

run_name=$1
checkpoint=$2
tag=$3
shift 3
log_dir=research/intent_phrase/logs
mkdir -p "$log_dir"
rm -f "$log_dir/${run_name}.done" "$log_dir/${run_name}.failed"
trap 'touch "$log_dir/${run_name}.failed"' ERR

.venv/bin/python scripts/plan_hierarchical.py \
  ckpt="$checkpoint" \
  device=cuda:0 \
  n_episodes=100 \
  seed=321 \
  hydra.run.dir="runs/${run_name}" \
  out="runs/${run_name}/${tag}.json" \
  "$@"

rm -f "$log_dir/${run_name}.failed"
touch "$log_dir/${run_name}.done"
