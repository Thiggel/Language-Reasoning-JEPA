#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" || -z "${TEXTJEPA_ROOT:-}" ]]; then
  echo "RUN_DIR and TEXTJEPA_ROOT must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
checkpoint=${2:?checkpoint path}
device=${DEVICE:-cuda:0}
if [[ ! -s "$checkpoint" ]]; then
  echo "checkpoint is missing or empty: $checkpoint" >&2
  exit 2
fi

# Evaluation runs after training in a separate bounded job so an expensive
# full-catalogue diagnostic cannot invalidate an otherwise complete model.
for interface in full oracle_feasible; do
  out="$RUN_DIR/${interface}_metrics.json"
  "$python_bin" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
    --kind jepa --checkpoint "$checkpoint" --device "$device" \
    --split val --episodes 500 --excess-actions 0 1 2 4 \
    --candidate-interface "$interface" --seed 7321 \
    --jepa-candidate-mode full --simulation-depth 4 --beam-width 4 \
    --out "$out"
done
