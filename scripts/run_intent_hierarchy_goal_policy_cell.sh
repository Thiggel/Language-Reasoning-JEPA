#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 NAME ALT_K SUBGOAL_RANK_WEIGHT"
  exit 2
fi

name=$1
alt_k=$2
rank_weight=$3
log_dir=research/intent_phrase/logs
mkdir -p "$log_dir"
rm -f "$log_dir/${name}.done" "$log_dir/${name}.failed"
trap 'touch "$log_dir/${name}.failed"' ERR

.venv/bin/python scripts/train.py \
  +experiment=intent_hier_repair \
  run_name="$name" \
  model.d_macro=32 \
  data.macro_alt_k="$alt_k" \
  train.train_low_value_head=true \
  objective.value.weight=1.0 \
  objective.macro_cf_dynamics.weight=1.0 \
  objective.macro_state_value.weight=0.25 \
  objective.macro_action_value.weight=0.25 \
  objective.macro_advantage_rank.weight=0.25 \
  objective.macro_receding_value.weight=1.0 \
  objective.macro_receding_value.discount=0.5 \
  objective.macro_receding_rank.weight=1.0 \
  objective.macro_receding_rank.discount=0.5 \
  objective.macro_support.weight=0.1 \
  objective.subgoal_action_rank.weight="$rank_weight"

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

.venv/bin/python scripts/audit_value_switch.py \
  --ckpt "runs/$name/best.pt" \
  --out "runs/$name/value_switch_audit.json" \
  --device cuda:0 --episodes 400

.venv/bin/python scripts/audit_subgoal_policy.py \
  --ckpt "runs/$name/best.pt" \
  --out "runs/$name/subgoal_policy_audit.json" \
  --device cuda:0 --examples 2000

rm -f "$log_dir/${name}.failed"
touch "$log_dir/${name}.done"
