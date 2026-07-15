#!/usr/bin/env bash
# Immutable direct worker for one completed token-prior checkpoint.
set -euo pipefail
NAME="${1:?cell name}"
SEED="${2:-0}"
ROOT="${TEXTJEPA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${TEXTJEPA_PYTHON:-$ROOT/.venv/bin/python}"
RUN_DIR="$ROOT/runs/overnight_token_prior_${NAME}_s${SEED}"
CKPT="$RUN_DIR/best.pt"
[[ -s "$CKPT" ]] || { echo "missing $CKPT" >&2; exit 2; }
rm -f "$RUN_DIR/DIAGNOSTICS_COMPLETE" "$RUN_DIR/DIAGNOSTICS_FAILED"
cd "$ROOT"
if ! {
  "$PYTHON" scripts/audit_token_prior.py --ckpt "$CKPT" --device cuda:0 \
    --examples 512 > "$RUN_DIR/token_prior_calibration.log" 2>&1
  "$PYTHON" scripts/audit_token_hierarchy_drift.py --ckpt "$CKPT" \
    --device cuda:0 --examples 256 --max-horizon 16 \
    > "$RUN_DIR/predictor_drift_curves.log" 2>&1
  "$PYTHON" scripts/probe_token_hierarchy_symbolic.py --ckpt "$CKPT" \
    --device cuda:0 --examples 512 > "$RUN_DIR/symbolic_linear_probes.log" 2>&1
}; then
  touch "$RUN_DIR/DIAGNOSTICS_FAILED"
  exit 1
fi
touch "$RUN_DIR/DIAGNOSTICS_COMPLETE"
