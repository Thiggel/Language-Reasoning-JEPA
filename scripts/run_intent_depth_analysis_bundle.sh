#!/usr/bin/env bash
# Run the planning-depth causal audit first, then fill missing five-seed GAR audits.
set -euo pipefail

[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2
  exit 2
}
python_bin=${1:?python executable}
full_checkpoint=${2:?full GAR checkpoint}
direct_checkpoint=${3:?direct-ranker checkpoint}
shift 3
device=${DEVICE:-cuda:0}
depth_episodes=${DEPTH_EPISODES:-150}
geometry_episodes=${GEOMETRY_EPISODES:-200}

for specification in "full:$full_checkpoint" "direct:$direct_checkpoint"; do
  label=${specification%%:*}
  checkpoint=${specification#*:}
  mkdir -p "$RUN_DIR/depth_$label"
  RUN_DIR="$RUN_DIR/depth_$label" DEVICE="$device" EPISODES="$depth_episodes" \
    bash "$TEXTJEPA_ROOT/scripts/run_intent_planning_depth_audit.sh" \
      "$python_bin" "$checkpoint"
done

index=0
for checkpoint in "$@"; do
  mkdir -p "$RUN_DIR/geometry_$index"
  RUN_DIR="$RUN_DIR/geometry_$index" DEVICE="$device" EPISODES="$geometry_episodes" \
    bash "$TEXTJEPA_ROOT/scripts/run_intent_decision_audit_cell.sh" \
      "$python_bin" jepa "$checkpoint"
  index=$((index + 1))
done
