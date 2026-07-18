#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?checkpoint path}
examples=${3:-32}
max_candidates=${4:-256}
max_steps=${5:-32}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
"$python_bin" "${TEXTJEPA_ROOT}/scripts/plan_faithful_token_edits.py" \
  --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
  --examples "$examples" --max-candidates "$max_candidates" \
  --max-steps "$max_steps" --out "$RUN_DIR/planning_metrics.json"
