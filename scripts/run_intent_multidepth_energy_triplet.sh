#!/usr/bin/env bash
# Run MSE-only, rank-only, and combined cells under one matched allocation.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
py=${1:?python executable}
dynamics=${2:?energy_only|dense_dynamics}
geometry=${3:?geometry mode}
lr=${4:-3e-4}
parent=$RUN_DIR
for loss in mse rank both; do
  export RUN_DIR="$parent/$loss"
  mkdir -p "$RUN_DIR"
  bash "$TEXTJEPA_ROOT/scripts/run_intent_multidepth_energy_cell.sh" \
    "$py" "$loss" "$dynamics" "$geometry" "$lr"
done
