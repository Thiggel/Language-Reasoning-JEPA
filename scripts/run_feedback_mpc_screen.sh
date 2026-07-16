#!/usr/bin/env bash
# Usage: run_feedback_mpc_screen.sh WORKER GPU
set -u
WORKER="${1:?worker 0..2}"
GPU="${2:?physical GPU}"
cd /vol/home-vol2/ml/laitenbf/TextJEPA || exit 1
CKPT=runs/hard_hier_v2_l2_s8_32_d32_16_reachhist_s0/best.pt
LOGDIR=runs/feedback_mpc_screen
mkdir -p "$LOGDIR"
FAIL="$LOGDIR/worker${WORKER}_failures.txt"
: > "$FAIL"

cells=()
for support in unconstrained conditional_bank conditional_prior; do
  for reach in 0 1; do
    cells+=("${support}|${reach}|boundary|0.5")
    cells+=("${support}|${reach}|l1_feedback|0.5")
    for threshold in 0.3 0.5 0.7; do
      cells+=("${support}|${reach}|adaptive|${threshold}")
    done
  done
done

for index in "${!cells[@]}"; do
  (( index % 3 == WORKER )) || continue
  IFS='|' read -r support reach feedback threshold <<< "${cells[$index]}"
  reach_arg=(); [[ "$reach" == 1 ]] && reach_arg=(--reachability-refine)
  tag="feedback_${feedback}_t${threshold}_e3"
  name="${support}_reach${reach}_${feedback}_t${threshold}"
  echo "START $name $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/plan_token_hierarchy_oracle_cem.py \
    --ckpt "$CKPT" --device cuda:0 --support-mode "$support" \
    --feedback-mode "$feedback" --feedback-threshold "$threshold" \
    --episodes 3 --max-tokens 96 --high-horizon 2 \
    --macro-candidates 256 --macro-iterations 5 --macro-elites 32 \
    --token-candidates 128 --token-iterations 4 --token-elites 16 \
    --reach-topn 8 --reach-budget-scale 0.25 \
    --bank-examples 256 --bank-size 2048 --output-tag "$tag" \
    "${reach_arg[@]}" > "$LOGDIR/$name.log" 2>&1
  status=$?
  [[ $status == 0 ]] || echo "$name status=$status" >> "$FAIL"
  echo "END $name status=$status $(date --iso-8601=seconds)"
done
touch "$LOGDIR/worker${WORKER}.complete"
