#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 NAME SEED {regression|pairwise|listwise|receding|combined|ood}"
  exit 2
fi

name=$1
seed=$2
mode=$3
weights=(
  objective.macro_action_value.weight=0.0
  objective.macro_advantage_rank.weight=0.0
  objective.macro_top1_rank.weight=0.0
  objective.macro_receding_value.weight=0.0
  objective.macro_receding_rank.weight=0.0
  objective.macro_ood_value_rank.weight=0.0
)
case "$mode" in
  regression)
    weights+=(objective.macro_action_value.weight=1.0)
    ;;
  pairwise)
    weights+=(objective.macro_advantage_rank.weight=1.0)
    ;;
  listwise)
    weights+=(objective.macro_top1_rank.weight=1.0)
    ;;
  receding)
    weights+=(
      objective.macro_receding_value.weight=1.0
      objective.macro_receding_rank.weight=1.0
    )
    ;;
  combined)
    weights+=(
      objective.macro_action_value.weight=0.25
      objective.macro_advantage_rank.weight=0.25
      objective.macro_receding_value.weight=1.0
      objective.macro_receding_rank.weight=1.0
    )
    ;;
  ood)
    weights+=(
      objective.macro_action_value.weight=1.0
      objective.macro_advantage_rank.weight=1.0
      objective.macro_ood_value_rank.weight=1.0
    )
    ;;
  *)
    echo "unknown value-head mode: $mode"
    exit 2
    ;;
esac

.venv/bin/python scripts/train.py \
  +experiment=intent_hier_repair \
  run_name="$name" \
  seed="$seed" \
  model.d_macro=32 \
  train.init_ckpt=runs/intent_hgoalpolicy_set_a8_w1/selected_planning.pt \
  train.init_mode=full \
  train.reset_macro_value_head=true \
  train.freeze_low_level=true \
  train.train_high_level=false \
  train.train_macro_value_head=true \
  train.epochs=5 \
  train.warmup_steps=100 \
  train.eval_batches=8 \
  data.train_size=20000 \
  data.val_size=1000 \
  data.macro_alt_k=8 \
  objective.hierarchy.weight=0.0 \
  objective.macro_prior.weight=0.0 \
  objective.macro_cf_dynamics.weight=0.0 \
  objective.macro_state_value.weight=0.0 \
  objective.macro_support.weight=0.0 \
  objective.subgoal_action_rank.weight=0.0 \
  "${weights[@]}"

.venv/bin/python scripts/audit_macro_heads.py \
  --ckpt "runs/$name/best.pt" \
  --out "runs/$name/macro_heads_audit.json" \
  --device cuda:0 --anchors 200 --max-candidates 128
