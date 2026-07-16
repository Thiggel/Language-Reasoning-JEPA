#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RUN_NAME D_MACRO N_LAYERS RESIDUAL"
  exit 2
fi

name=$1
d_macro=$2
layers=$3
residual=$4
log_dir=research/intent_phrase/logs
mkdir -p "$log_dir"
rm -f "$log_dir/${name}.done" "$log_dir/${name}.failed"
trap 'touch "$log_dir/${name}.failed"' ERR

.venv/bin/python scripts/train.py \
  +experiment=intent_hier_screen \
  run_name="$name" \
  model.d_macro="$d_macro" \
  model.macro_variational=false \
  model.high_predictor_kind=causal \
  model.high_predictor_layers="$layers" \
  model.high_predictor_residual="$residual"

.venv/bin/python scripts/audit_intent_hierarchy.py \
  --ckpt "runs/$name/best.pt" \
  --out "runs/$name/hierarchy_dynamics_audit.json" \
  --device cuda:0 \
  --examples 640

rm -f "$log_dir/${name}.failed"
touch "$log_dir/${name}.done"
