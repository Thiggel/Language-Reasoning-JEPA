#!/usr/bin/env bash
set -euo pipefail
python_bin=${1:?python executable}
seed=${2:?seed}
shift 2
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
export TMPDIR="/tmp/tj-${RUN_ID:-edit-mdlm-$$}"
mkdir -p "$TMPDIR"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/train_edit_mdlm.py" \
  --out "$RUN_DIR" --device "${DEVICE:-cuda:0}" --seed "$seed" "$@"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_edit_mdlm.py" \
  --ckpt "$RUN_DIR/model/best.pt" --out "$RUN_DIR/metrics.json" \
  --device "${DEVICE:-cuda:0}" --examples 128 --batches 16
