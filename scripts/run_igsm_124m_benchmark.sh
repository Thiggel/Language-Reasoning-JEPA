#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
kind=${2:?model kind}
microbatch=${3:?microbatch size}
: "${RUN_DIR:?RUN_DIR must be supplied by researchctl}"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/benchmark_igsm_124m_step.py" \
  "$kind" --batch-size "$microbatch" --warmup 1 --steps 3 \
  --output "$RUN_DIR/benchmark.json"
