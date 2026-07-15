#!/usr/bin/env bash
# Train one exploratory token-prior hierarchy cell and run matched no-LM plans.
# Usage: run_token_prior_overnight_cell.sh NAME PRIOR_W HIDDEN DETACH SMOOTH \
#        SPANS DIMS PHASES LOW_DENSE HIGH_DENSE [TRAIN_SIZE] [EPOCHS] [SEED]
set -euo pipefail

NAME="${1:?name}"
PRIOR_WEIGHT="${2:?prior objective weight}"
HIDDEN="${3:?prior hidden width}"
DETACH="${4:?detach prior state}"
SMOOTH="${5:?label smoothing}"
SPANS="${6:?level spans}"
DIMS="${7:?level dims}"
PHASES="${8:?phase augmentation flags}"
LOW_DENSE="${9:?low dense depth}"
HIGH_DENSE="${10:?high dense depth}"
TRAIN_SIZE="${11:-6000}"
EPOCHS="${12:-3}"
SEED="${13:-0}"

ROOT="${TEXTJEPA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${TEXTJEPA_PYTHON:-$ROOT/.venv/bin/python}"
RUN_NAME="overnight_token_prior_${NAME}_s${SEED}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
LOG_DIR="$ROOT/runs/overnight_token_prior_logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"
rm -f "$RUN_DIR/COMPLETE" "$RUN_DIR/FAILED"

cd "$ROOT"
if ! "$PYTHON" scripts/train_token_hierarchy_v2.py \
  +experiment=hard_hier_v2_screen \
  run_name="$RUN_NAME" seed="$SEED" \
  data.train_size="$TRAIN_SIZE" data.val_size=512 \
  train.epochs="$EPOCHS" train.batch_size=32 train.num_workers=4 \
  train.eval_batches=16 train.log_every=25 \
  model.level_spans="$SPANS" model.level_dims="$DIMS" \
  model.variational_levels='[false]' model.phase_augmented_levels="$PHASES" \
  model.low_dense_depth="$LOW_DENSE" model.high_dense_depth="$HIGH_DENSE" \
  model.use_token_prior=true model.token_prior_hidden="$HIDDEN" \
  model.token_prior_detach_state="$DETACH" \
  objective.token_prior="$PRIOR_WEIGHT" \
  objective.token_prior_label_smoothing="$SMOOTH" \
  > "$LOG_DIR/${NAME}_train.log" 2>&1; then
  touch "$RUN_DIR/FAILED"
  exit 1
fi

CKPT="$RUN_DIR/best.pt"
COMMON=(
  --ckpt "$CKPT" --device cuda:0 --support-mode conditional_bank
  --feedback-mode adaptive --feedback-threshold 0.5
  --token-execution-chunk 1 --episodes 3 --max-tokens 64
  --high-horizon 2 --macro-candidates 512 --macro-iterations 8
  --macro-elites 64 --token-candidates 256 --token-iterations 5
  --token-elites 32 --bank-examples 256 --bank-size 2048
)

for SPEC in \
  'prior_greedy|0.0|1.0|0' \
  'prior_shooting|0.0|0.8|32' \
  'prior_shooting|0.0|1.0|0' \
  'prior_energy|0.1|1.0|0' \
  'prior_energy|0.5|1.0|0' \
  'prior_energy|1.0|1.0|0'
do
  IFS='|' read -r MODE PLAN_WEIGHT TEMP TOPK <<< "$SPEC"
  TAG="${NAME}_${MODE}_w${PLAN_WEIGHT}_t${TEMP}_k${TOPK}"
  "$PYTHON" scripts/plan_token_hierarchy_oracle_cem.py \
    "${COMMON[@]}" --token-proposal "$MODE" \
    --token-prior-weight "$PLAN_WEIGHT" \
    --token-prior-temperature "$TEMP" --token-prior-topk "$TOPK" \
    --output-tag "$TAG" > "$LOG_DIR/${TAG}.log" 2>&1
done

# The most promising support-aware proposal is also tested with lower-level
# reachability feedback to isolate the two planner interventions.
TAG="${NAME}_shooting_reach"
"$PYTHON" scripts/plan_token_hierarchy_oracle_cem.py \
  "${COMMON[@]}" --token-proposal prior_shooting \
  --token-prior-temperature 0.8 --token-prior-topk 32 \
  --reachability-refine --reach-topn 16 --reach-budget-scale 0.25 \
  --output-tag "$TAG" > "$LOG_DIR/${TAG}.log" 2>&1

touch "$RUN_DIR/COMPLETE"
