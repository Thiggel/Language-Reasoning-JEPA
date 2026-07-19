#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
condition=${2:?experiment config}
seed=${3:?seed}
shift 3
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
export TMPDIR="/tmp/tj-${RUN_ID:-token-edit-$$}"
mkdir -p "$TMPDIR"
model_dir="$RUN_DIR/model"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/train.py" \
  "+experiment=$condition" "hydra.run.dir=$model_dir" \
  "run_name=${RUN_ID:-${condition}_s${seed}}" "seed=$seed" "device=${DEVICE:-cuda:0}" \
  "$@"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_faithful_token_edits.py" \
  --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
  --examples 256 --corruption-mode mixed --out "$RUN_DIR/metrics.json"
for corruption_mode in mask replace remove; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_faithful_token_edits.py" \
    --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
    --examples 256 --corruption-mode "$corruption_mode" \
    --out "$RUN_DIR/metrics_${corruption_mode}.json"
done
