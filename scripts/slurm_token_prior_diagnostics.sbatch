#!/usr/bin/env bash
set -euo pipefail
: "${TEXTJEPA_ROOT:?}"
: "${TEXTJEPA_PYTHON:?}"
OFFSET="${MATRIX_OFFSET:-0}"
SEED="${EXPERIMENT_SEED:-1}"
INDEX=$((OFFSET + SLURM_ARRAY_TASK_ID))
LINE=$(sed -n "$((INDEX + 2))p" \
  "$TEXTJEPA_ROOT/research/hard_text/overnight_token_prior_matrix.tsv")
IFS=$'\t' read -r NAME _ <<< "$LINE"
RUN_DIR="$TEXTJEPA_ROOT/runs/overnight_token_prior_${NAME}_s${SEED}"
CKPT="$RUN_DIR/best.pt"
[[ -s "$CKPT" ]] || { echo "missing $CKPT" >&2; exit 2; }

if [[ "$TEXTJEPA_ROOT" == /home/atuin/* ]]; then
  module load cuda/12.8.1 >/dev/null 2>&1 || true
elif [[ "$TEXTJEPA_ROOT" == /scratch/* ]]; then
  module unload sw.clx.el9 sw.a100.el9 sw.pvc sw.pvc.el9 sw.genoa.el9 \
    >/dev/null 2>&1 || true
  module load sw.a100
fi
export PYTHONPATH="$TEXTJEPA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$TEXTJEPA_ROOT"
"$TEXTJEPA_PYTHON" scripts/audit_token_prior.py \
  --ckpt "$CKPT" --device cuda:0 --examples 512 \
  > "$RUN_DIR/token_prior_calibration.log" 2>&1
"$TEXTJEPA_PYTHON" scripts/audit_token_hierarchy_drift.py \
  --ckpt "$CKPT" --device cuda:0 --examples 256 --max-horizon 16 \
  > "$RUN_DIR/predictor_drift_curves.log" 2>&1
"$TEXTJEPA_PYTHON" scripts/probe_token_hierarchy_symbolic.py \
  --ckpt "$CKPT" --device cuda:0 --examples 512 \
  > "$RUN_DIR/symbolic_linear_probes.log" 2>&1
touch "$RUN_DIR/DIAGNOSTICS_COMPLETE"
