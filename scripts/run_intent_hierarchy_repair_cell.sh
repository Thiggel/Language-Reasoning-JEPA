#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 || $# -gt 10 ]]; then
  echo "usage: $0 NAME D_MACRO CF_DYN STATE_V ACTION_Q RANK SUPPORT ALT_K [REACH] [SUPPORT_SCALES]"
  exit 2
fi

name=$1
d_macro=$2
cf_dyn=$3
state_v=$4
action_q=$5
rank=$6
support=$7
alt_k=$8
reach=${9:-0}
support_scales=${10:-[3.0]}
log_dir=research/intent_phrase/logs
mkdir -p "$log_dir"
rm -f "$log_dir/${name}.done" "$log_dir/${name}.failed"
trap 'touch "$log_dir/${name}.failed"' ERR

.venv/bin/python scripts/train.py \
  +experiment=intent_hier_repair \
  run_name="$name" \
  model.d_macro="$d_macro" \
  model.macro_support_scales="$support_scales" \
  data.macro_alt_k="$alt_k" \
  objective.macro_cf_dynamics.weight="$cf_dyn" \
  objective.macro_state_value.weight="$state_v" \
  objective.macro_action_value.weight="$action_q" \
  objective.macro_advantage_rank.weight="$rank" \
  objective.macro_support.weight="$support" \
  objective.hierarchy_reachability.weight="$reach"

.venv/bin/python scripts/audit_intent_hierarchy.py \
  --ckpt "runs/$name/best.pt" \
  --out "runs/$name/hierarchy_dynamics_audit.json" \
  --device cuda:0 \
  --examples 640

.venv/bin/python scripts/audit_macro_heads.py \
  --ckpt "runs/$name/best.pt" \
  --out "runs/$name/macro_heads_audit.json" \
  --device cuda:0 \
  --anchors 200 \
  --max-candidates 128

rm -f "$log_dir/${name}.failed"
touch "$log_dir/${name}.done"
