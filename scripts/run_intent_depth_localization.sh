#!/usr/bin/env bash
# Paired fixed-checkpoint localization of intent-planning depth failure.
set -euo pipefail

[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2
  exit 2
}
python_bin=${1:?python executable}
checkpoint=${2:?checkpoint path}
label=${3:?checkpoint label}
device=${DEVICE:-cuda:0}
episodes=${EPISODES:-300}
diagnostic_episodes=${DIAGNOSTIC_EPISODES:-100}
diagnostic_cap=${DIAGNOSTIC_CAP:-128}
seed=${SEED:-8051}

[[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }
sha256sum "$checkpoint" > "$RUN_DIR/checkpoint.sha256"
"$python_bin" "$TEXTJEPA_ROOT/scripts/audit_intent_depth_localization.py" \
  --checkpoint "$checkpoint" \
  --device "$device" \
  --episodes "$episodes" \
  --diagnostic-episodes "$diagnostic_episodes" \
  --diagnostic-cap "$diagnostic_cap" \
  --seed "$seed" \
  --out "$RUN_DIR/depth_localization.json"

"$python_bin" - "$RUN_DIR/depth_localization.json" "$RUN_DIR/metrics.json" "$label" <<'PY'
import json
import sys

source, destination, label = sys.argv[1:]
payload = json.load(open(source))
summary = {
    "label": label,
    "protocol": payload["protocol"],
    "checkpoint": payload["checkpoint"],
    "episodes": payload["episodes"],
    "diagnostic_episodes": payload["diagnostic_episodes"],
    "information_boundary": payload["information_boundary"],
    "policy_cells": payload["policy_cells"],
    "diagnostics": payload["diagnostics"],
}
with open(destination, "w") as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")
PY
