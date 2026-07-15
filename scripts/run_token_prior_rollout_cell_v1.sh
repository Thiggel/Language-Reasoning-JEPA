#!/usr/bin/env bash
# Immutable predicted-state token-prior training/evaluation worker.
set -euo pipefail
ROW_INDEX="${1:?zero-based matrix row}"
SEED="${2:-0}"
ROOT="${TEXTJEPA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${TEXTJEPA_PYTHON:-$ROOT/.venv/bin/python}"
LINE=$(sed -n "$((ROW_INDEX + 2))p" \
  "$ROOT/research/hard_text/token_prior_rollout_matrix.tsv")
IFS=$'\t' read -r ROW_NAME PRIOR_WEIGHT ROLLOUT_WEIGHT DETACH DENSE DISCOUNT \
  HIDDEN SMOOTH PURPOSE <<< "$LINE"
NAME="rollout_${ROW_NAME}"
RUN_NAME="overnight_token_prior_${NAME}_s${SEED}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
LOG_DIR="$ROOT/runs/overnight_token_prior_logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"
rm -f "$RUN_DIR/COMPLETE" "$RUN_DIR/FAILED"
cd "$ROOT"
echo "cell=$NAME seed=$SEED purpose=$PURPOSE"

if ! "$PYTHON" scripts/train_token_hierarchy_v2.py \
  +experiment=hard_hier_v2_screen run_name="$RUN_NAME" seed="$SEED" \
  data.train_size=6000 data.val_size=512 train.epochs=3 \
  train.batch_size=32 train.num_workers=4 train.eval_batches=16 \
  train.log_every=25 model.level_spans='[8,32]' model.level_dims='[32,16]' \
  model.variational_levels='[false]' \
  model.phase_augmented_levels='[false,true]' \
  model.low_dense_depth="$DENSE" model.high_dense_depth="$DENSE" \
  model.use_token_prior=true model.token_prior_hidden="$HIDDEN" \
  model.token_prior_detach_state="$DETACH" \
  objective.token_prior="$PRIOR_WEIGHT" \
  objective.token_prior_rollout="$ROLLOUT_WEIGHT" \
  objective.token_prior_rollout_discount="$DISCOUNT" \
  objective.token_prior_label_smoothing="$SMOOTH" \
  > "$LOG_DIR/${NAME}_s${SEED}_train.log" 2>&1; then
  touch "$RUN_DIR/FAILED"
  exit 1
fi

"$ROOT/scripts/run_token_prior_eval_cell_v1.sh" "$NAME" "$SEED"
"$ROOT/scripts/run_token_prior_diagnostics_v1.sh" "$NAME" "$SEED"
touch "$RUN_DIR/COMPLETE"
