#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
condition=${2:?experiment config}
seed=${3:?seed}
name="${condition}_s${seed}"
model_dir="$RUN_DIR/model"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/train.py" \
  "+experiment=$condition" "hydra.run.dir=$model_dir" \
  "run_name=$name" "seed=$seed" "device=${DEVICE:-cuda:0}"

PY="$python_bin" bash "${TEXTJEPA_ROOT}/scripts/eval_run.sh" \
  "$model_dir" "${DEVICE:-cuda:0}"

jq -n \
  --slurpfile strict "$model_dir/plan_slack0_look1.json" \
  --slurpfile slack "$model_dir/plan_slack2_look1.json" \
  '{strict: $strict[0], slack2: $slack[0]}' > "$RUN_DIR/metrics.json"
