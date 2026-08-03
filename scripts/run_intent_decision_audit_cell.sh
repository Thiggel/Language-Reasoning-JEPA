#!/usr/bin/env bash
# Frozen-checkpoint decision and GAR geometry audit. No training.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo 'RUN_DIR and TEXTJEPA_ROOT required' >&2
  exit 2
}
py=${1:?python executable}
kind=${2:?jepa, token_lm, or sentence_lm}
checkpoint=${3:?checkpoint path}
episodes=${EPISODES:-500}
seed=${SEED:-7321}
device=${DEVICE:-cuda:0}
for split in train val; do
  common=(
    --checkpoint "$checkpoint" --split "$split" --episodes "$episodes"
    --seed "$seed" --device "$device" --forced-errors
    --out "$RUN_DIR/${split}_audit.json"
    --features-out "$RUN_DIR/${split}_features.npz"
  )
  if [[ "$kind" == jepa ]]; then
    "$py" "$TEXTJEPA_ROOT/scripts/audit_intent_gar_geometry.py" "${common[@]}"
  else
    "$py" "$TEXTJEPA_ROOT/scripts/audit_intent_lm_decisions.py" \
      --kind "$kind" "${common[@]}"
  fi
done
