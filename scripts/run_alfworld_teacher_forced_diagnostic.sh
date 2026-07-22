#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" || -z "${TEXTJEPA_ROOT:-}" ]]; then
  echo "RUN_DIR and TEXTJEPA_ROOT must be supplied by researchctl" >&2
  exit 2
fi
python_bin=${1:?python}
shared_root=${TEXTJEPA_SHARED_ROOT:-/vol/home-vol2/ml/laitenbf/TextJEPA}
source_round=$shared_root/runs/autonomy/intent_phrase/2026-07-22-intent-alfworld-overfit-gate-recovery-v2
export ALFWORLD_PILOT_ROOT=${ALFWORLD_PILOT_ROOT:-$shared_root/data/intent_phrase/alfworld/pilot_v1}
export PYTHONPATH=$TEXTJEPA_ROOT/src

"$python_bin" "$TEXTJEPA_ROOT/scripts/diagnose_observed_action_teacher_forcing.py" \
  --device "${DEVICE:-cuda:0}" --split train --episodes 8 \
  --top-m 1 4 8 16 32 64 --beam-width 4 \
  --checkpoint "$source_round/intent-alfworld-overfit-lr3e-4-recovery-v2/model/last.pt" \
  --checkpoint "$source_round/intent-alfworld-overfit-lr1e-3-recovery-v2/model/last.pt" \
  --checkpoint "$source_round/intent-alfworld-overfit-lr3e-3-recovery-v2/model/last.pt" \
  --out "$RUN_DIR/metrics.json"
