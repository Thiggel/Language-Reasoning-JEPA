#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
condition=${2:?experiment config}
seed=${3:?seed}
examples=${4:-128}
max_candidates=${5:-256}
max_steps=${6:-4}
shift 6

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

bash "${TEXTJEPA_ROOT}/scripts/run_faithful_token_edit_cell.sh" \
  "$python_bin" "$condition" "$seed" "$@"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/plan_faithful_token_edits.py" \
  --ckpt "$RUN_DIR/model/best.pt" --device "${DEVICE:-cuda:0}" \
  --examples "$examples" --max-candidates "$max_candidates" \
  --max-steps "$max_steps" --out "$RUN_DIR/planning_metrics.json"
