#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

root=$RUN_DIR
run_gate() {
  local label=$1 method=$2
  shift 2
  mkdir -p "$root/$label"
  RUN_DIR="$root/$label" RUN_ID="${RUN_ID}-$label" \
    bash "${TEXTJEPA_ROOT}/scripts/run_edit_b256_speed_gate.sh" \
      "$python_bin" "$method" "$@"
}

run_gate mdlm mdlm 16 32 64 128 256
run_gate token edit_igsm_original_token 32 64 128 256
run_gate sentence edit_igsm_original_sentence 8 16 32 64 128 256
run_gate token_sentence edit_igsm_original_token_sentence 8 16 32 64 128 256

"$python_bin" - "$root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
payload = {}
for name in ("mdlm", "token", "sentence", "token_sentence"):
    payload[name] = json.loads((root / name / "metrics.json").read_text())
(root / "metrics.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
