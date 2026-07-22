#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
method=${2:?mdlm or Hydra experiment name}
shift 2

export EFFECTIVE_BATCH=256
export THROUGHPUT_STEPS=${THROUGHPUT_STEPS:-12}
export EDIT_EMA_VICREG=1

exec bash "${TEXTJEPA_ROOT}/scripts/run_original_edit_throughput_ladder.sh" \
  "$python_bin" "$method" "$@"
