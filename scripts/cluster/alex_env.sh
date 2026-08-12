#!/usr/bin/env bash
# Sourced by every Alex (NHR@FAU) cell script.  Establishes the module
# environment, outbound proxy, caches, and temporary directories documented in
# docs/clusters/ALEX.md.  Requires SNAP (immutable code snapshot) to be set.
: "${SNAP:?export SNAP to the immutable code snapshot directory}"
: "${TEXTJEPA_AUTONOMY_ROOT:?export TEXTJEPA_AUTONOMY_ROOT}"

module load cuda/12.8.1 >/dev/null 2>&1 || true
export http_proxy="${http_proxy:-http://proxy:80}"
export https_proxy="${https_proxy:-http://proxy:80}"

export TEXTJEPA_ROOT="$SNAP"
export PYTHONPATH="$SNAP/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$TEXTJEPA_AUTONOMY_ROOT/.cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export HF_HOME="$XDG_CACHE_HOME/hf"
# Node-local scratch per Alex policy; never shared across array tasks.
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}/tj-${SLURM_JOB_ID:-$$}}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME" "$TMPDIR"
chmod 700 "$TMPDIR" || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
