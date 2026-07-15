#!/usr/bin/env bash
# Generic Slurm array worker; cluster-specific resources are supplied by sbatch.
set -euo pipefail

: "${TEXTJEPA_ROOT:?export TEXTJEPA_ROOT to the scratch/project clone}"
: "${TEXTJEPA_PYTHON:?export TEXTJEPA_PYTHON to a compatible shared environment}"
OFFSET="${MATRIX_OFFSET:-0}"
SEED="${EXPERIMENT_SEED:-1}"
INDEX=$((OFFSET + SLURM_ARRAY_TASK_ID))
MATRIX="$TEXTJEPA_ROOT/research/hard_text/overnight_token_prior_matrix.tsv"
LINE=$(sed -n "$((INDEX + 2))p" "$MATRIX")
if [[ -z "$LINE" ]]; then
  echo "No matrix row at zero-based index $INDEX" >&2
  exit 2
fi
IFS=$'\t' read -r \
  NAME WEIGHT HIDDEN DETACH SMOOTH SPANS DIMS PHASES LOW_DENSE HIGH_DENSE PURPOSE \
  <<< "$LINE"

if [[ "$TEXTJEPA_ROOT" == /home/atuin/* ]]; then
  module load cuda/12.8.1 >/dev/null 2>&1 || true
  export http_proxy="${http_proxy:-http://proxy:80}"
  export https_proxy="${https_proxy:-http://proxy:80}"
elif [[ "$TEXTJEPA_ROOT" == /scratch/* ]]; then
  module unload sw.clx.el9 sw.a100.el9 sw.pvc sw.pvc.el9 sw.genoa.el9 \
    >/dev/null 2>&1 || true
  module load sw.a100
fi

export PYTHONPATH="$TEXTJEPA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$TEXTJEPA_ROOT/.cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TMPDIR="${SLURM_TMPDIR:-$TEXTJEPA_ROOT/tmp}"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" \
  "$TEXTJEPA_ROOT/runs/slurm_logs"

echo "cluster_host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
echo "cell=$NAME seed=$SEED purpose=$PURPOSE"
echo "python=$TEXTJEPA_PYTHON root=$TEXTJEPA_ROOT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

exec "$TEXTJEPA_ROOT/scripts/run_token_prior_overnight_cell.sh" \
  "$NAME" "$WEIGHT" "$HIDDEN" "$DETACH" "$SMOOTH" "$SPANS" "$DIMS" \
  "$PHASES" "$LOW_DENSE" "$HIGH_DENSE" 6000 3 "$SEED"
