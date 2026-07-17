#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?checkpoint path}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
for corruption_mode in mixed mask replace remove; do
  output="$RUN_DIR/metrics_${corruption_mode}.json"
  [[ "$corruption_mode" == mixed ]] && output="$RUN_DIR/metrics.json"
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_faithful_token_edits.py" \
    --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" --examples 256 \
    --corruption-mode "$corruption_mode" --out "$output"
done
