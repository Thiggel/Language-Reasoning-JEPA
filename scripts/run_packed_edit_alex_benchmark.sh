#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

"$python_bin" "$TEXTJEPA_ROOT/scripts/audit_packed_sequence_gpu.py" \
  --out "$RUN_DIR/correctness"

export THROUGHPUT_STEPS=${THROUGHPUT_STEPS:-12}
parent_run_dir=$RUN_DIR
parent_run_id=${RUN_ID:-packed-edit-benchmark}

RUN_DIR="$parent_run_dir/ema" RUN_ID="${parent_run_id}-ema" \
  bash "$TEXTJEPA_ROOT/scripts/run_original_edit_throughput_ladder.sh" \
  "$python_bin" edit_igsm_original_token_packed 32 64 128 256 512

RUN_DIR="$parent_run_dir/noema_visreg" RUN_ID="${parent_run_id}-noema-visreg" \
  bash "$TEXTJEPA_ROOT/scripts/run_original_edit_throughput_ladder.sh" \
  "$python_bin" edit_igsm_original_token_noema_visreg_packed 16 32 64 128 256 512

"$python_bin" - "$parent_run_dir" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
payload = {
    "correctness": json.loads((root / "correctness/metrics.json").read_text()),
    "ema": json.loads((root / "ema/metrics.json").read_text()),
    "noema_visreg": json.loads(
        (root / "noema_visreg/metrics.json").read_text()
    ),
}
(root / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
