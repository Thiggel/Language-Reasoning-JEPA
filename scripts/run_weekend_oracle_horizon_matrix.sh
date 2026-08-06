#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
parent=${RUN_DIR:?RUN_DIR must be supplied by researchctl}

run_cell() {
  local horizon=$1
  local name="oracle-beam-gar-h${horizon}"
  mkdir -p "$parent/$name"
  RUN_DIR="$parent/$name" bash scripts/run_corrected_gar_search_cell.sh \
    "$python_bin" "$name" 0 0.25 false 1 0.3 "$horizon" \
    combined 0.3 prior oracle_beam 4 4 1 2 2 1000 1 8
}

run_cell 1
run_cell 2
run_cell 4
run_cell 8

"$python_bin" - "$parent" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("*/metrics.json")):
    summary[path.parent.name] = json.loads(path.read_text())
(root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
