#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
source_model=${2:?source model directory}
checkpoint_label=${3:?checkpoint label}
model_dir="$RUN_DIR/model"

export TMPDIR="/tmp/tj-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"

if [[ ! -f "$source_model/best.pt" ]]; then
  echo "missing source checkpoint: $source_model/best.pt" >&2
  exit 3
fi
mkdir -p "$model_dir"
cp "$source_model/best.pt" "$model_dir/best.pt"

PY="$python_bin" bash "${TEXTJEPA_ROOT}/scripts/eval_run.sh" \
  "$model_dir" "${DEVICE:-cuda:0}"

jq -n \
  --arg checkpoint_label "$checkpoint_label" \
  --arg source_model "$source_model" \
  --slurpfile strict "$model_dir/plan_slack0_look1.json" \
  --slurpfile slack "$model_dir/plan_slack2_look1.json" \
  '{"label": $checkpoint_label, source_model: $source_model,
    checkpoint_status: "best checkpoint from walltime-truncated training",
    strict: $strict[0], slack2: $slack[0]}' \
  > "$RUN_DIR/metrics.json"
