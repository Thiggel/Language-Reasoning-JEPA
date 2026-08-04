#!/usr/bin/env bash
# Fixed-checkpoint test-time-scaling evaluation for intent-phrase JEPA.
set -euo pipefail

[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2
  exit 2
}
python_bin=${1:?python executable}
checkpoint=${2:?checkpoint path}
device=${DEVICE:-cuda:0}
episodes=${N_EPISODES:-200}
seed=${SEED:-321}
# depth:max_expand. Depth > 1 is deliberately candidate-privileged because
# the reference graph supplies future feasible actions.
cells=${SCALING_CELLS:-"1:64 2:64 4:64 8:64 2:4 2:8 4:4 4:8"}

[[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }
sha256sum "$checkpoint" > "$RUN_DIR/checkpoint.sha256"

for cell in $cells; do
  depth=${cell%%:*}
  cap=${cell##*:}
  case "$depth:$cap:$episodes:$seed" in
    *[!0-9:]*) echo "invalid scaling cell or numeric setting: $cell" >&2; exit 2;;
  esac
  oracle=false
  label=common-current-menu
  if (( depth > 1 )); then
    oracle=true
    label=symbolic-future-action-tree
  fi
  for slack in 0 2; do
    stem="$RUN_DIR/depth${depth}_cap${cap}_slack${slack}"
    "$python_bin" "$TEXTJEPA_ROOT/scripts/plan.py" \
      "ckpt=$checkpoint" "device=$device" split=val \
      "n_episodes=$episodes" "seed=$seed" "slack=$slack" \
      "lookahead=$depth" "max_expand=$cap" \
      "allow_oracle_future_actions=$oracle" energy=value \
      measure_flops=true "out=${stem}.json" "compute_out=${stem}_compute.json"
  done
done

"$python_bin" - "$RUN_DIR" "$checkpoint" "$cells" "$episodes" "$seed" <<'PY'
import hashlib, json, pathlib, sys

run = pathlib.Path(sys.argv[1])
checkpoint = pathlib.Path(sys.argv[2])
cells = sys.argv[3].split()
summary = {
    "protocol": "fixed-checkpoint-symbolic-candidate-tree",
    "candidate_privilege": {
        "depth_1": "common currently feasible action menu",
        "depth_gt_1": "reference-graph future feasible action tree",
    },
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "n_episodes": int(sys.argv[4]),
    "seed": int(sys.argv[5]),
    "cells": {},
}
for cell in cells:
    depth, cap = map(int, cell.split(":"))
    key = f"depth{depth}_cap{cap}"
    summary["cells"][key] = {}
    for slack in (0, 2):
        stem = run / f"{key}_slack{slack}"
        metrics = json.loads(stem.with_suffix(".json").read_text())
        compute = json.loads((run / f"{key}_slack{slack}_compute.json").read_text())
        if compute["checkpoint_sha256"] != summary["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed during evaluation at {key}")
        summary["cells"][key][str(slack)] = {
            "metrics": metrics,
            "compute": compute,
        }
(run / "scaling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
