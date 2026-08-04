#!/usr/bin/env bash
# Fixed-checkpoint causal audit of planning-depth degradation.
set -euo pipefail

[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2
  exit 2
}
python_bin=${1:?python executable}
checkpoint=${2:?checkpoint path}
device=${DEVICE:-cuda:0}
episodes=${EPISODES:-250}
seed=${SEED:-7411}
cells=${DEPTH_AUDIT_CELLS:-"1:64 2:4 2:8 2:16 2:32 2:64 4:4 4:8 4:16 4:32 4:64"}

[[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }
sha256sum "$checkpoint" > "$RUN_DIR/checkpoint.sha256"
read -r -a cell_args <<< "$cells"
"$python_bin" "$TEXTJEPA_ROOT/scripts/audit_intent_planning_depth.py" \
  --checkpoint "$checkpoint" --device "$device" --episodes "$episodes" \
  --seed "$seed" --cells "${cell_args[@]}" --out "$RUN_DIR/depth_audit.json"
