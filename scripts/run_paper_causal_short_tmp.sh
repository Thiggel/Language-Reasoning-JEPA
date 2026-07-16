#!/usr/bin/env bash
set -euo pipefail

# Direct Grünau controller paths can exceed Linux's AF_UNIX socket limit when
# PyTorch data-loader workers derive their socket directory from RUN_DIR.
# Keep each job's multiprocessing directory short and unique.
short_id=${RUN_ID:-direct-$$}
export TMPDIR="/tmp/tj-${short_id}"
mkdir -p "$TMPDIR"

bash "${TEXTJEPA_ROOT}/scripts/run_paper_causal_single_cell.sh" "$@"
