#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
kind=${2:?model kind}
shift 2
: "${RUN_DIR:?RUN_DIR must be supplied by researchctl}"

for microbatch in "$@"; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/benchmark_igsm_124m_step.py" \
    "$kind" --batch-size "$microbatch" --warmup 1 --steps 3 \
    --accumulation 1 \
    --output "$RUN_DIR/benchmark_mb${microbatch}.json"
done

"$python_bin" - "$RUN_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("benchmark_mb*.json")):
    item = json.loads(path.read_text())
    rows.append({
        "microbatch": item["microbatch"],
        "status": item["status"],
        "peak_allocated_gib": item.get("peak_allocated_gib"),
        "examples_per_second": item.get("examples_per_second"),
    })
(root / "capacity_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
PY
