#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 NAME DENSE_DEPTH LOWER_ENDPOINT_WEIGHT"
  exit 2
fi

name=$1
dense_depth=$2
lower_endpoint=$3
dense_weight=0.0
one_step_weight=0.0
train_predictor=false
if [[ "$dense_depth" -gt 0 ]]; then
  dense_weight=1.0
  train_predictor=true
fi
log_dir=research/intent_phrase/logs
mkdir -p "$log_dir"
rm -f "$log_dir/${name}.done" "$log_dir/${name}.failed"
trap 'touch "$log_dir/${name}.failed"' ERR

.venv/bin/python scripts/train.py \
  +experiment=intent_hier_repair \
  run_name="$name" \
  model.d_macro=32 \
  model.dense_rollout_depth="$dense_depth" \
  data.macro_alt_k=8 \
  train.train_low_predictor="$train_predictor" \
  train.train_low_value_head=true \
  objective.value.weight=1.0 \
  objective.latent_pred.weight="$one_step_weight" \
  objective.dense_rollout.weight="$dense_weight" \
  objective.dense_rollout.horizon_discount=0.7 \
  objective.macro_cf_dynamics.weight=1.0 \
  objective.macro_state_value.weight=0.25 \
  objective.macro_action_value.weight=0.25 \
  objective.macro_advantage_rank.weight=0.25 \
  objective.macro_receding_value.weight=1.0 \
  objective.macro_receding_value.discount=0.5 \
  objective.macro_receding_rank.weight=1.0 \
  objective.macro_receding_rank.discount=0.5 \
  objective.macro_support.weight=0.1 \
  objective.lower_hierarchy_rollout.weight="$lower_endpoint"

.venv/bin/python scripts/audit_intent_hierarchy.py \
  --ckpt "runs/$name/best.pt" \
  --out "runs/$name/hierarchy_dynamics_audit.json" \
  --device cuda:0 --examples 640

.venv/bin/python scripts/audit_macro_heads.py \
  --ckpt "runs/$name/best.pt" \
  --out "runs/$name/macro_heads_audit.json" \
  --device cuda:0 --anchors 200 --max-candidates 128

.venv/bin/python scripts/audit_hierarchy_retrieval.py \
  --ckpt "runs/$name/best.pt" \
  --out "runs/$name/hierarchy_retrieval_audit.json" \
  --device cuda:0 --anchors 200 --max-candidates 128

rm -f "$log_dir/${name}.failed"
touch "$log_dir/${name}.done"
