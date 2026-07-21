#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
seed=${2:?seed}
lr=${3:?learning rate}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
export TMPDIR="/tmp/tj-${RUN_ID:-edit-mdlm-confidence-$$}"
mkdir -p "$TMPDIR"

"$python_bin" "$TEXTJEPA_ROOT/scripts/train_edit_mdlm.py" \
  --out "$RUN_DIR" --device "${DEVICE:-cuda:0}" --seed "$seed" \
  --train-size 2000 --val-size 256 --epochs 4 --batch-size 8 \
  --d-model 320 --layers 8 --heads 8 --lr "$lr" \
  --warmup-steps 100 --eval-batches 8

"$python_bin" "$TEXTJEPA_ROOT/scripts/audit_edit_mdlm.py" \
  --ckpt "$RUN_DIR/model/best.pt" --out "$RUN_DIR/metrics.json" \
  --device "${DEVICE:-cuda:0}" --examples 8 --batches 1
