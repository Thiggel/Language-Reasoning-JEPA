#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
condition=${2:?experiment config}
seed=${3:?seed}
learning_rate=${4:?learning rate}
epochs=${5:?epochs}
warmup_steps=${6:?warmup steps}
name="${condition}_lr${learning_rate}_e${epochs}_s${seed}"
model_dir="$RUN_DIR/model"

export TMPDIR="/tmp/tj-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/train.py" \
  "+experiment=$condition" \
  "hydra.run.dir=$model_dir" \
  "run_name=$name" \
  "seed=$seed" \
  "train.lr=$learning_rate" \
  "train.epochs=$epochs" \
  "train.warmup_steps=$warmup_steps" \
  "device=${DEVICE:-cuda:0}"

PY="$python_bin" bash "${TEXTJEPA_ROOT}/scripts/eval_run.sh" \
  "$model_dir" "${DEVICE:-cuda:0}"

jq -n \
  --arg condition "$condition" \
  --arg learning_rate "$learning_rate" \
  --argjson epochs "$epochs" \
  --argjson warmup_steps "$warmup_steps" \
  --slurpfile strict "$model_dir/plan_slack0_look1.json" \
  --slurpfile slack "$model_dir/plan_slack2_look1.json" \
  '{condition: $condition, learning_rate: $learning_rate, epochs: $epochs,
    warmup_steps: $warmup_steps, strict: $strict[0], slack2: $slack[0]}' \
  > "$RUN_DIR/metrics.json"
