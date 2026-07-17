#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?checkpoint path}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
"$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_faithful_token_edits.py" \
  --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" --examples 256 \
  --corruption-mode mixed --component-falsifiers --out "$RUN_DIR/metrics.json"
