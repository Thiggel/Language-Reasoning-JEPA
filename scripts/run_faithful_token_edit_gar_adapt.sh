#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?checkpoint path}
regime=${3:?replay or teacher}
seed=${4:-0}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

"$python_bin" "${TEXTJEPA_ROOT}/scripts/finetune_faithful_token_edit_gar.py" \
  --ckpt "$checkpoint" --regime "$regime" --seed "$seed" \
  --device "${DEVICE:-cuda:0}" --out-dir "$RUN_DIR"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/plan_faithful_token_edits.py" \
  --ckpt "$RUN_DIR/model/best.pt" --device "${DEVICE:-cuda:0}" \
  --examples 64 --max-candidates 256 --max-steps 8 \
  --out "$RUN_DIR/planning_metrics.json"
