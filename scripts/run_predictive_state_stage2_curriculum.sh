#!/usr/bin/env bash
set -euo pipefail

# The Stage 2 horizon curriculum, run as one chained cell.
#
# Each stage resumes from the previous stage's checkpoint, so 1 -> 4 -> 8 -> 16
# -> 32 is a real curriculum rather than five independent restarts. Generated
# prefix replay switches on from horizon 16, where open-loop drift starts to
# matter and teacher actions stop being representative of what the jump model
# will actually condition on.

: "${RUN_DIR:?RUN_DIR must identify the immutable run cell}"
: "${PREDICTIVE_STATE_TOKEN_BLOCKS:?shared token-block file is required}"
: "${PREDICTIVE_STATE_STAGE1_CHECKPOINT:?a Stage 1 checkpoint is required}"
[[ -s "$PREDICTIVE_STATE_STAGE1_CHECKPOINT" ]] || {
  echo "missing Stage 1 checkpoint: $PREDICTIVE_STATE_STAGE1_CHECKPOINT" >&2
  exit 2
}
python_bin=${TEXTJEPA_PYTHON:-/vol/home-vol2/ml/laitenbf/TextJEPA/.venv/bin/python}

horizons=(${PREDICTIVE_STATE_HORIZONS:-1 4 8 16 32})
steps=(${PREDICTIVE_STATE_STAGE_STEPS:-600 600 600 600 1000})
replay=(${PREDICTIVE_STATE_REPLAY_FRACTIONS:-0.0 0.0 0.0 0.25 0.25})
[[ ${#horizons[@]} -eq ${#steps[@]} && ${#horizons[@]} -eq ${#replay[@]} ]] || {
  echo "curriculum arrays differ in length" >&2
  exit 2
}

checkpoint="$PREDICTIVE_STATE_STAGE1_CHECKPOINT"
for index in "${!horizons[@]}"; do
  horizon=${horizons[$index]}
  stage_dir="$RUN_DIR/h${horizon}"
  mkdir -p "$stage_dir"
  echo "=== horizon $horizon from $checkpoint" >&2
  "$python_bin" scripts/train_action_transition_rollout.py \
    --checkpoint "$checkpoint" \
    --token-blocks "$PREDICTIVE_STATE_TOKEN_BLOCKS" \
    --output "$stage_dir" \
    --horizon "$horizon" \
    --steps "${steps[$index]}" \
    --on-policy-fraction "${replay[$index]}" \
    --microbatch-size "${PREDICTIVE_STATE_MICROBATCH:-4}" \
    --gradient-accumulation "${PREDICTIVE_STATE_ACCUMULATION:-2}" \
    --sequence-length "${PREDICTIVE_STATE_SEQUENCE_LENGTH:-512}" \
    --predictor-learning-rate "${PREDICTIVE_STATE_PREDICTOR_LR:-1e-4}" \
    --lora-learning-rate "${PREDICTIVE_STATE_BACKBONE_LR:-5e-5}" \
    --dtype "${PREDICTIVE_STATE_DTYPE:-bfloat16}" --device cuda --seed 0
  checkpoint="$stage_dir/last.pt"
done

# The gate is measured on the final curriculum checkpoint, at horizons well
# past the longest one trained, so the claim is about generalization rather
# than about the training horizon.
"$python_bin" scripts/evaluate_action_transition_rollout.py \
  --checkpoint "$checkpoint" \
  --token-blocks "$PREDICTIVE_STATE_TOKEN_BLOCKS" \
  --output "$RUN_DIR/rollout_evaluation.json" \
  --horizon 1 --horizon 2 --horizon 4 --horizon 8 --horizon 16 \
  --horizon 32 --horizon 64 --horizon 128 --horizon 256 \
  --batches "${PREDICTIVE_STATE_EVAL_BATCHES:-8}"
cp "$RUN_DIR/rollout_evaluation.json" "$RUN_DIR/run_summary.json"
