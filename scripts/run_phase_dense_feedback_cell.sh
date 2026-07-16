#!/usr/bin/env bash
# Usage: script NAME GPU LOW_DEPTH HIGH_DEPTH DISCOUNT
set -u
NAME="${1:?name}" GPU="${2:?gpu}" LOW="${3:?low depth}" HIGH="${4:?high depth}" DISCOUNT="${5:?discount}"
cd /vol/home-vol2/ml/laitenbf/TextJEPA || exit 1
RUN="hard_hier_v2_l2_phase_${NAME}_s0"
LOGDIR=runs/phase_dense_feedback_screen
mkdir -p "$LOGDIR"

CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/train_token_hierarchy_v2.py \
  data=igsm_hard +experiment=hard_hier_v2_screen run_name="$RUN" \
  seed=0 device=cuda:0 data.train_size=6000 data.val_size=500 \
  train.epochs=3 train.eval_batches=15 \
  model.level_spans='[8,32]' model.level_dims='[32,16]' \
  model.variational_levels='[false,false]' \
  model.phase_augmented_levels='[false,true]' \
  model.low_dense_depth="$LOW" model.high_dense_depth="$HIGH" \
  objective.dense_discount="$DISCOUNT" > "$LOGDIR/${NAME}_train.log" 2>&1
status=$?
if [[ $status != 0 ]]; then echo "train status=$status" > "$LOGDIR/${NAME}.failed"; exit $status; fi
CKPT="runs/$RUN/best.pt"

CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/audit_token_hierarchy_drift.py \
  --ckpt "$CKPT" --device cuda:0 --examples 128 --max-horizon 16 \
  > "$LOGDIR/${NAME}_drift.log" 2>&1

for feedback in boundary l1_feedback adaptive; do
  for reach in 0 1; do
    extra=(); [[ "$reach" == 1 ]] && extra=(--reachability-refine)
    CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/plan_token_hierarchy_oracle_cem.py \
      --ckpt "$CKPT" --device cuda:0 --support-mode conditional_bank \
      --feedback-mode "$feedback" --feedback-threshold 0.5 \
      --episodes 3 --max-tokens 96 --high-horizon 2 \
      --macro-candidates 256 --macro-iterations 5 --macro-elites 32 \
      --token-candidates 128 --token-iterations 4 --token-elites 16 \
      --reach-topn 8 --reach-budget-scale 0.25 \
      --bank-examples 256 --bank-size 2048 \
      --output-tag "phase_${NAME}_${feedback}_e3" "${extra[@]}" \
      > "$LOGDIR/${NAME}_${feedback}_reach${reach}.log" 2>&1
  done
done
touch "$LOGDIR/${NAME}.complete"
