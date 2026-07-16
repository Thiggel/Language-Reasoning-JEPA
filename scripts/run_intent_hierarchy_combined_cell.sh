#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 NAME DENSE_DEPTH DISCOUNT LOWER_ENDPOINT_WEIGHT REACH_WEIGHT"
  exit 2
fi

name=$1
dense_depth=$2
discount=$3
lower_endpoint=$4
reach=$5
dense_weight=0.0
one_step_weight=1.0
if [[ "$dense_depth" -gt 0 ]]; then
  dense_weight=1.0
  one_step_weight=0.0
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
  train.train_low_predictor=true \
  objective.latent_pred.weight="$one_step_weight" \
  objective.dense_rollout.weight="$dense_weight" \
  objective.dense_rollout.horizon_discount=0.7 \
  objective.macro_cf_dynamics.weight=1.0 \
  objective.macro_state_value.weight=0.25 \
  objective.macro_action_value.weight=0.25 \
  objective.macro_advantage_rank.weight=0.25 \
  objective.macro_receding_value.weight=1.0 \
  objective.macro_receding_value.discount="$discount" \
  objective.macro_receding_rank.weight=1.0 \
  objective.macro_receding_rank.discount="$discount" \
  objective.macro_support.weight=0.1 \
  objective.lower_hierarchy_rollout.weight="$lower_endpoint" \
  objective.hierarchy_reachability.weight="$reach"

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
