#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?checkpoint path}
method=${3:?mdlm or jepa}
examples=${4:-8}
shift 4 || true
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "checkpoint not found: $checkpoint" >&2
  exit 3
fi

"$python_bin" "$TEXTJEPA_ROOT/scripts/eval_original_igsm_lengths.py" \
  --method "$method" --ckpt "$checkpoint" \
  --device "${DEVICE:-cuda:0}" --examples-per-cell "$examples" \
  --out "$RUN_DIR/length_metrics.json" "$@"
