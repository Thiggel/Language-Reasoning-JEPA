#!/usr/bin/env bash
# Evaluate already-trained non-oracle ALFWorld checkpoints over compute depth.
set -euo pipefail

if [[ -z "${RUN_DIR:-}" || -z "${TEXTJEPA_ROOT:-}" ]]; then
  echo "RUN_DIR and TEXTJEPA_ROOT must be supplied by researchctl" >&2
  exit 2
fi
python_bin=${1:?python executable}
checkpoint=${2:?checkpoint}
device=${DEVICE:-cuda:0}

for depth in 1 2 4 8; do
  for beam in 1 4 8; do
    "$python_bin" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
      --kind jepa --checkpoint "$checkpoint" --device "$device" \
      --split train --episodes 8 --excess-actions 0 1 2 4 \
      --simulation-depth "$depth" --jepa-candidate-mode full \
      --beam-width "$beam" \
      --out "$RUN_DIR/depth${depth}_beam${beam}_train.json"
    "$python_bin" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
      --kind jepa --checkpoint "$checkpoint" --device "$device" \
      --split val --episodes 4 --excess-actions 0 1 2 4 \
      --simulation-depth "$depth" --jepa-candidate-mode full \
      --beam-width "$beam" \
      --out "$RUN_DIR/depth${depth}_beam${beam}_val.json"
  done
done
