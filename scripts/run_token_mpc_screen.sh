#!/usr/bin/env bash
# Usage: script WORKER GPU (four workers)
set -u
WORKER="${1:?worker}" GPU="${2:?gpu}"
cd /vol/home-vol2/ml/laitenbf/TextJEPA || exit 1
LOGDIR=runs/token_mpc_screen
mkdir -p "$LOGDIR"
: > "$LOGDIR/worker${WORKER}_failures.txt"

cells=(
  'n2|runs/hard_hier_v2_l2_phase_n2_s0/best.pt|boundary|0|0'
  'n2|runs/hard_hier_v2_l2_phase_n2_s0/best.pt|boundary|1|0'
  'n2|runs/hard_hier_v2_l2_phase_n2_s0/best.pt|adaptive|0|0'
  'n2|runs/hard_hier_v2_l2_phase_n2_s0/best.pt|adaptive|1|0'
  'n4|runs/hard_hier_v2_l2_phase_n4_s0/best.pt|adaptive|1|0'
  'd4_l05|runs/hard_hier_v2_l2_phase_d4_l05_s0/best.pt|adaptive|0|0'
  'd4_l05|runs/hard_hier_v2_l2_phase_d4_l05_s0/best.pt|adaptive|1|0'
  'nonphase|runs/hard_hier_v2_l2_s8_32_d32_16_reachhist_s0/best.pt|adaptive|0|0'
  'nonphase|runs/hard_hier_v2_l2_s8_32_d32_16_reachhist_s0/best.pt|adaptive|1|0'
  'flat_h8|runs/hard_hier_v2_flat_control_s0/best.pt|boundary|0|8'
  'flat_h32|runs/hard_hier_v2_flat_control_s0/best.pt|boundary|0|32'
)

for index in "${!cells[@]}"; do
  (( index % 4 == WORKER )) || continue
  IFS='|' read -r name ckpt feedback reach flat_h <<< "${cells[$index]}"
  reach_arg=(); [[ "$reach" == 1 ]] && reach_arg=(--reachability-refine)
  flat_arg=(); [[ "$flat_h" != 0 ]] && flat_arg=(--flat --flat-horizon "$flat_h")
  tag="tokenmpc_${name}_${feedback}_reach${reach}_e3"
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/plan_token_hierarchy_oracle_cem.py \
    --ckpt "$ckpt" --device cuda:0 --support-mode conditional_bank \
    --feedback-mode "$feedback" --feedback-threshold 0.5 \
    --token-execution-chunk 1 --episodes 3 --max-tokens 96 \
    --macro-candidates 256 --macro-iterations 5 --macro-elites 32 \
    --token-candidates 128 --token-iterations 4 --token-elites 16 \
    --reach-topn 8 --reach-budget-scale 0.25 \
    --bank-examples 256 --bank-size 2048 --output-tag "$tag" \
    "${reach_arg[@]}" "${flat_arg[@]}" > "$LOGDIR/${tag}.log" 2>&1
  status=$?
  [[ $status == 0 ]] || echo "$tag status=$status" >> "$LOGDIR/worker${WORKER}_failures.txt"
done
touch "$LOGDIR/worker${WORKER}.complete"
