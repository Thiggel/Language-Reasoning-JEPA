#!/usr/bin/env bash
set -euo pipefail

name=${1:-intent_action_support}
state_mode=${2:-true}
log_dir=research/intent_phrase/logs
mkdir -p "$log_dir"
rm -f "$log_dir/${name}.done" "$log_dir/${name}.failed"
trap 'touch "$log_dir/${name}.failed"' ERR

.venv/bin/python scripts/train.py \
  +experiment=intent_action_support \
  run_name="$name" \
  model.action_support_states="$state_mode"

rm -f "$log_dir/${name}.failed"
touch "$log_dir/${name}.done"
