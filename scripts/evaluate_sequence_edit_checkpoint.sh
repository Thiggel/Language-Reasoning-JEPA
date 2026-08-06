#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?checkpoint to evaluate}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

export TMPDIR="/tmp/tj-${RUN_ID:-edit-eval-$$}"
mkdir -p "$TMPDIR"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_faithful_token_edits.py" \
  --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
  --examples 256 --out "$RUN_DIR/metrics.json"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/plan_faithful_token_edits.py" \
  --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
  --examples 32 --max-candidates 256 --max-steps 32 \
  --out "$RUN_DIR/planning_metrics.json"
