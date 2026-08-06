#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
parent=${RUN_DIR:?RUN_DIR must be supplied by researchctl}

run_cell() {
  local name=$1 regression=$2 pairwise=$3 k=$4 batch=$5
  mkdir -p "$parent/$name"
  RUN_DIR="$parent/$name" bash scripts/run_corrected_gar_search_cell.sh \
    "$python_bin" "$name" 0 "$regression" false "$pairwise" 0.3 1 \
    combined 0.3 prior sampled 4 "$k" 1 4 4 2000 2 "$batch"
}

run_cell mse-only-prior-k16 1 0 16 8
run_cell rank-mse025-prior-k16 0.25 1 16 8
run_cell mse-only-prior-k32 1 0 32 6

"$python_bin" - "$parent" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("*/metrics.json")):
    summary[path.parent.name] = json.loads(path.read_text())
(root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
