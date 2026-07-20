#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?checkpoint}
horizon=${3:?horizon}
beam_width=${4:-8}
prior_weight=${5:-0.05}
gar_weight=${6:-1.0}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
"$python_bin" "${TEXTJEPA_ROOT}/scripts/plan_token_refinement_mpc.py" \
  --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" --examples 32 \
  --horizon "$horizon" --beam-width "$beam_width" \
  --top-positions 4 --top-tokens 4 --max-candidates 16 --max-steps 32 \
  --prior-weight "$prior_weight" --gar-weight "$gar_weight" \
  --out "$RUN_DIR/planning_metrics.json"
