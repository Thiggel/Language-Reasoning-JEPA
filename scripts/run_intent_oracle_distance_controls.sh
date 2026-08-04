#!/usr/bin/env bash
# Matched oracle-distance controls for fixed-checkpoint planning depth.
set -euo pipefail

[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2
  exit 2
}
python_bin=${1:?python executable}
checkpoint=${2:?checkpoint path}

RUN_DIR="$RUN_DIR" \
N_EPISODES="${N_EPISODES:-500}" \
SEED="${SEED:-7321}" \
SCALING_CELLS="${ORACLE_CELLS:-1:64 2:8 2:64 4:8 4:64}" \
SCALING_CONTROLS="oracle_goal symbolic_oracle_goal symbolic_exact_distance" \
SCALING_SLACKS="0" \
DEVICE="${DEVICE:-cuda:0}" \
bash "$TEXTJEPA_ROOT/scripts/run_intent_fixed_checkpoint_scaling.sh" \
  "$python_bin" "$checkpoint"
