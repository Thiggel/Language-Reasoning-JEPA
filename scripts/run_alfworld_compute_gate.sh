#!/usr/bin/env bash
# Evaluate already-trained non-oracle ALFWorld checkpoints over compute depth.
set -euo pipefail

if [[ -z "${RUN_DIR:-}" || -z "${TEXTJEPA_ROOT:-}" ]]; then
  echo "RUN_DIR and TEXTJEPA_ROOT must be supplied by researchctl" >&2
  exit 2
fi
python_bin=${1:?python executable}
checkpoint=${2:?checkpoint}
requested_depth=${3:-all}
device=${DEVICE:-cuda:0}
shared_root=${TEXTJEPA_SHARED_ROOT:-/vol/home-vol2/ml/laitenbf/TextJEPA}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$shared_root/data/intent_phrase/alfworld_engine}
eval_python=${ALFWORLD_PYTHON:-/vol/home-vol2/ml/laitenbf/.venv-alfworld/bin/python}
project_site=${TEXTJEPA_PROJECT_SITE:-$shared_root/.venv/lib64/python3.11/site-packages}
export PYTHONPATH=$TEXTJEPA_ROOT/src:$project_site

depths=(1 2 4 8)
if [[ "$requested_depth" != all ]]; then
  depths=("$requested_depth")
fi
for depth in "${depths[@]}"; do
  for beam in 1 4 8; do
    "$eval_python" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
      --kind jepa --checkpoint "$checkpoint" --device "$device" \
      --split train --episodes 8 --excess-actions 0 1 2 4 \
      --simulation-depth "$depth" --jepa-candidate-mode full \
      --beam-width "$beam" \
      --out "$RUN_DIR/depth${depth}_beam${beam}_train.json"
    "$eval_python" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
      --kind jepa --checkpoint "$checkpoint" --device "$device" \
      --split val --episodes 4 --excess-actions 0 1 2 4 \
      --simulation-depth "$depth" --jepa-candidate-mode full \
      --beam-width "$beam" \
      --out "$RUN_DIR/depth${depth}_beam${beam}_val.json"
  done
done
