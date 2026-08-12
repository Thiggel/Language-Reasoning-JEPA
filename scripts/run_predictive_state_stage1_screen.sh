#!/usr/bin/env bash
set -euo pipefail

# One Stage 1 upper-half LoRA screen cell. Unlike the frozen diagnostic, every
# cell reads one pre-built shared token-block file so all arms consume
# bit-identical train and validation tensors; the corpus is never rebuilt here.

variant=${1:?variant: full, no_action, action_only, same_layer, nitp, or ntp_only}
seed=${2:-0}
case "$variant" in
  full|no_action|action_only|same_layer|nitp|ntp_only) ;;
  *) echo "unsupported stage 1 variant: $variant" >&2; exit 2 ;;
esac
: "${RUN_DIR:?RUN_DIR must identify the immutable run cell}"
: "${PREDICTIVE_STATE_TOKEN_BLOCKS:?shared token-block file is required}"
[[ -s "$PREDICTIVE_STATE_TOKEN_BLOCKS" ]] || {
  echo "token-block file is missing: $PREDICTIVE_STATE_TOKEN_BLOCKS" >&2
  exit 2
}
python_bin=${TEXTJEPA_PYTHON:-/vol/home-vol2/ml/laitenbf/TextJEPA/.venv/bin/python}
mkdir -p "$RUN_DIR/model"

# An optional predictor bottleneck. Left unset, the predictor keeps its
# protocol width and can satisfy the auxiliary loss without the backbone.
bottleneck=()
if [[ -n "${PREDICTIVE_STATE_PROJECTION_SIZE:-}" ]]; then
  bottleneck+=(--projection-size "$PREDICTIVE_STATE_PROJECTION_SIZE")
fi
if [[ -n "${PREDICTIVE_STATE_ACTION_PROJECTION_SIZE:-}" ]]; then
  bottleneck+=(--action-projection-size "$PREDICTIVE_STATE_ACTION_PROJECTION_SIZE")
fi
if [[ -n "${PREDICTIVE_STATE_PREDICTOR_WIDTH:-}" ]]; then
  bottleneck+=(--predictor-width "$PREDICTIVE_STATE_PREDICTOR_WIDTH")
fi

# Layers 1-12 and the embeddings stay frozen, so the layer-12 prediction target
# is a fixed anchor. Only the source half (13-24) and the predictor adapt; the
# target cannot drift to meet the predictor.
"$python_bin" scripts/train_action_transition.py \
  --token-blocks "$PREDICTIVE_STATE_TOKEN_BLOCKS" \
  --output "$RUN_DIR/model" --variant "$variant" --backbone-mode lora \
  --steps "${PREDICTIVE_STATE_STEPS:-1220}" \
  --microbatch-size "${PREDICTIVE_STATE_MICROBATCH:-8}" \
  --gradient-accumulation "${PREDICTIVE_STATE_ACCUMULATION:-2}" \
  --sequence-length "${PREDICTIVE_STATE_SEQUENCE_LENGTH:-1024}" \
  --eval-every "${PREDICTIVE_STATE_EVAL_EVERY:-100}" \
  --eval-batches "${PREDICTIVE_STATE_EVAL_BATCHES:-16}" \
  --lora-rank 16 --lora-alpha 32 \
  --predictor-learning-rate "${PREDICTIVE_STATE_PREDICTOR_LR:-3e-4}" \
  --lora-learning-rate "${PREDICTIVE_STATE_LORA_LR:-1e-4}" \
  --prediction-weight "${PREDICTIVE_STATE_PREDICTION_WEIGHT:-0.1}" \
  --scale-weight "${PREDICTIVE_STATE_SCALE_WEIGHT:-0.01}" \
  "${bottleneck[@]+"${bottleneck[@]}"}" \
  --dtype "${PREDICTIVE_STATE_DTYPE:-bfloat16}" --device cuda --seed "$seed"

"$python_bin" scripts/summarize_predictive_state_stage1.py \
  --metrics "$RUN_DIR/model/metrics.json" \
  --output "$RUN_DIR/run_summary.json"
cp "$RUN_DIR/run_summary.json" "$RUN_DIR/eval_summary.json"
