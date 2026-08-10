#!/usr/bin/env bash
set -euo pipefail

variant=${1:?variant: full, no_action, or action_only}
seed=${2:-0}
case "$variant" in
  full|no_action|action_only) ;;
  *) echo "unsupported frozen diagnostic variant: $variant" >&2; exit 2 ;;
esac
: "${RUN_DIR:?RUN_DIR must identify the immutable run cell}"
root=${TEXTJEPA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
python_bin=${TEXTJEPA_PYTHON:-/vol/home-vol2/ml/laitenbf/TextJEPA/.venv/bin/python}
mkdir -p "$RUN_DIR/data" "$RUN_DIR/model"

"$python_bin" scripts/prepare_predictive_state_corpus.py \
  --output "$RUN_DIR/data/wikitext2_qwen_ctx256.pt" \
  --download-path "$RUN_DIR/data/wikitext2_train.txt" \
  --context-length 256 --validation-fraction 0.1 --seed "$seed"

"$python_bin" scripts/train_action_transition.py \
  --token-blocks "$RUN_DIR/data/wikitext2_qwen_ctx256.pt" \
  --output "$RUN_DIR/model" --variant "$variant" --backbone-mode frozen \
  --steps "${PREDICTIVE_STATE_STEPS:-200}" \
  --microbatch-size "${PREDICTIVE_STATE_MICROBATCH:-2}" \
  --gradient-accumulation 1 --sequence-length 256 \
  --eval-every "${PREDICTIVE_STATE_EVAL_EVERY:-100}" \
  --eval-batches "${PREDICTIVE_STATE_EVAL_BATCHES:-8}" \
  --predictor-learning-rate 3e-4 --prediction-weight 0.1 \
  --scale-weight 0.01 --dtype float16 --device cuda --seed "$seed"

"$python_bin" scripts/summarize_predictive_state_stage1.py \
  --metrics "$RUN_DIR/model/metrics.json" \
  --output "$RUN_DIR/run_summary.json"
cp "$RUN_DIR/run_summary.json" "$RUN_DIR/eval_summary.json"
