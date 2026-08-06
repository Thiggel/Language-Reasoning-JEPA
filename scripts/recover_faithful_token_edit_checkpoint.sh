#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?checkpoint path}
examples=${3:-128}
max_candidates=${4:-256}
max_steps=${5:-4}

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

export TMPDIR="/tmp/tj-${RUN_ID:-edit-recovery-$$}"
mkdir -p "$TMPDIR"

for corruption_mode in mixed mask replace remove; do
  suffix="_${corruption_mode}"
  if [[ "$corruption_mode" == mixed ]]; then
    suffix=""
  fi
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_faithful_token_edits.py" \
    --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
    --examples 256 --corruption-mode "$corruption_mode" \
    --out "$RUN_DIR/metrics${suffix}.json"
done

"$python_bin" "${TEXTJEPA_ROOT}/scripts/plan_faithful_token_edits.py" \
  --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
  --examples "$examples" --max-candidates "$max_candidates" \
  --max-steps "$max_steps" --out "$RUN_DIR/planning_metrics.json"
