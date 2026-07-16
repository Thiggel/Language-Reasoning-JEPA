#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
source_ckpt=${2:?existing checkpoint}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
if [[ ! -f "$source_ckpt" ]]; then
  echo "checkpoint not found: $source_ckpt" >&2
  exit 2
fi

model_dir="$RUN_DIR/model"
mkdir -p "$model_dir"
ln -s "$source_ckpt" "$model_dir/best.pt"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/probe_semantic_token_hierarchy.py" \
  --ckpt "$model_dir/best.pt" --device cuda:0 --examples 1000 --max-points 12000
"$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_token_hierarchy_gradients.py" \
  --ckpt "$model_dir/best.pt" --device cuda:0 --batch-size 16
"$python_bin" - "$model_dir" "$RUN_DIR/metrics.json" <<'PY'
import json, pathlib, sys
root, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
result = {}
for key, name in {
    "representation": "semantic_representation_probes.json",
    "gradients": "gradient_diagnostics.json",
}.items():
    result[key] = json.loads((root / name).read_text())
destination.write_text(json.dumps(result, indent=2) + "\n")
PY
