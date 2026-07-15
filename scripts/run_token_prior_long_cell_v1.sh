#!/usr/bin/env bash
set -euo pipefail
ROW_INDEX="${1:?zero-based row}"
ROOT="${TEXTJEPA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${TEXTJEPA_PYTHON:-$ROOT/.venv/bin/python}"
LINE=$(sed -n "$((ROW_INDEX + 2))p" \
  "$ROOT/research/hard_text/token_prior_long_matrix.tsv")
IFS=$'\t' read -r NAME SEED PRIOR ROLLOUT DETACH DENSE DISCOUNT HIDDEN SMOOTH \
  SPANS DIMS PHASES PURPOSE <<< "$LINE"
CELL="long_${NAME}"
RUN_NAME="overnight_token_prior_${CELL}_s${SEED}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
LOG_DIR="$ROOT/runs/overnight_token_prior_logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"
rm -f "$RUN_DIR/COMPLETE" "$RUN_DIR/FAILED"
cd "$ROOT"
echo "cell=$CELL seed=$SEED purpose=$PURPOSE"
if ! "$PYTHON" scripts/train_token_hierarchy_v2.py \
  +experiment=hard_hier_v2_screen run_name="$RUN_NAME" seed="$SEED" \
  data.train_size=12000 data.val_size=1000 train.epochs=8 \
  train.batch_size=32 train.num_workers=4 train.eval_batches=30 \
  train.log_every=50 model.level_spans="$SPANS" model.level_dims="$DIMS" \
  model.variational_levels='[false]' model.phase_augmented_levels="$PHASES" \
  model.low_dense_depth="$DENSE" model.high_dense_depth="$DENSE" \
  model.use_token_prior=true model.token_prior_hidden="$HIDDEN" \
  model.token_prior_detach_state="$DETACH" objective.token_prior="$PRIOR" \
  objective.token_prior_rollout="$ROLLOUT" \
  objective.token_prior_rollout_discount="$DISCOUNT" \
  objective.token_prior_label_smoothing="$SMOOTH" \
  > "$LOG_DIR/${CELL}_train.log" 2>&1; then
  touch "$RUN_DIR/FAILED"
  exit 1
fi
"$ROOT/scripts/run_token_prior_eval_cell_v1.sh" "$CELL" "$SEED"
"$ROOT/scripts/run_token_prior_diagnostics_v1.sh" "$CELL" "$SEED"
touch "$RUN_DIR/COMPLETE"
