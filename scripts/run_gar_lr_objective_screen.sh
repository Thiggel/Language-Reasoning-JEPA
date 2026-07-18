#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
parent=${RUN_DIR:?RUN_DIR must be supplied by researchctl}

run_cell() {
  local name=$1 regression=$2 pairwise=$3 lr=$4
  mkdir -p "$parent/$name"
  RUN_DIR="$parent/$name" bash scripts/run_corrected_gar_search_cell.sh \
    "$python_bin" "$name" 0 "$regression" false "$pairwise" 0.3 1 \
    combined 0.3 prior sampled 4 32 1 4 4 2000 2 6 32 16 pairwise 32 "$lr"
}

# Tune each objective on its own learning-rate curve. The rank+MSE 3e-4 cell
# reproduces the current recipe as an internal anchor.
for lr in 1e-4 3e-4 1e-3; do
  run_cell "mse-only-lr${lr}" 1 0 "$lr"
  run_cell "rank-mse025-lr${lr}" 0.25 1 "$lr"
done

"$python_bin" - "$parent" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("*/metrics.json")):
    summary[path.parent.name] = json.loads(path.read_text())
(root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
