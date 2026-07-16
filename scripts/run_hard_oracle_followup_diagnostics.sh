#!/usr/bin/env bash
set -u
GPU="${1:-0}"
cd /vol/home-vol2/ml/laitenbf/TextJEPA || exit 1
PRIMARY=runs/hard_oracle_cem_fast_logs/COMPLETE
while [[ ! -f "$PRIMARY" ]]; do sleep 30; done
LOGDIR=runs/hard_oracle_cem_fast_logs
for name in flat l1 l2 l3; do
  case "$name" in
    flat) ckpt=runs/hard_hier_v2_flat_control_s0/best.pt ;;
    l1) ckpt=runs/hard_hier_v2_l1_s8_d32_s0/best.pt ;;
    l2) ckpt=runs/hard_hier_v2_l2_s8_32_d32_16_s0/best.pt ;;
    l3) ckpt=runs/hard_hier_v2_l3_s8_32_96_d32_16_8_s0/best.pt ;;
  esac
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/audit_token_planning_interfaces.py \
    --ckpt "$ckpt" --device cuda:0 --examples 64 --horizons 1 4 8 16 \
    --candidates 256 --iterations 5 > "$LOGDIR/${name}_planning_interfaces.log" 2>&1
done
touch "$LOGDIR/FOLLOWUP_COMPLETE"
