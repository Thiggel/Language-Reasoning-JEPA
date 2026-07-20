#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
seed=${2:?seed}
shift 2
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
export TMPDIR="/tmp/tj-${RUN_ID:-token-refinement-$$}"
mkdir -p "$TMPDIR"
model_dir="$RUN_DIR/model"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/train.py" \
  "+experiment=edit_token_refinement_gar_positive_anchor" \
  "hydra.run.dir=$model_dir" "run_name=${RUN_ID}" \
  "seed=$seed" "device=${DEVICE:-cuda:0}" "$@"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_faithful_token_edits.py" \
  --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
  --examples 64 --corruption-mode iterative_refinement \
  --out "$RUN_DIR/metrics.json"
