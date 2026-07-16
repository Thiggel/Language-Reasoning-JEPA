#!/usr/bin/env bash
# Full post-training evaluation for one run: probes, planning, geometry plots,
# and (when enabled in the checkpoint config) GAR teacher quality.
# Usage: scripts/eval_run.sh runs/disc_base [cuda:0]
set -euo pipefail
RUN=$1
DEV=${2:-cuda:0}
PY=${PY:-.venv2/bin/python}

$PY scripts/probe.py ckpt="$RUN/best.pt" device="$DEV"
$PY scripts/plan.py ckpt="$RUN/best.pt" device="$DEV" slack=0
$PY scripts/plan.py ckpt="$RUN/best.pt" device="$DEV" slack=2
$PY scripts/analyze.py ckpt="$RUN/best.pt" device="$DEV"

# Exit status 0 means this checkpoint was trained with GAR candidates.  Read
# only the stored config here so ordinary/non-discourse runs remain cheap.
if $PY - "$RUN/best.pt" <<'PY'
import sys
import torch

cfg = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["cfg"]
raise SystemExit(0 if int(cfg.get("data", {}).get("geo_rank_k", 0)) > 0 else 1)
PY
then
  $PY scripts/audit_gar_teacher.py \
    --ckpt "$RUN/best.pt" --device "$DEV" --n-anchors 100
fi
