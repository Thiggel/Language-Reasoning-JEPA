#!/usr/bin/env bash
# Full post-training evaluation for one run: probes, planning, geometry plots.
# Usage: scripts/eval_run.sh runs/disc_base [cuda:0]
set -euo pipefail
RUN=$1
DEV=${2:-cuda:0}
PY=${PY:-.venv2/bin/python}

$PY scripts/probe.py ckpt="$RUN/best.pt" device="$DEV"
$PY scripts/plan.py ckpt="$RUN/best.pt" device="$DEV" slack=0
$PY scripts/plan.py ckpt="$RUN/best.pt" device="$DEV" slack=2
$PY scripts/analyze.py ckpt="$RUN/best.pt" device="$DEV"
