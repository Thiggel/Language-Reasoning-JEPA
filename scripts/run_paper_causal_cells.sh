#!/usr/bin/env bash
# Run explicitly assigned condition:seed cells on one GPU. This is used to
# borrow a free GPU on another shared-filesystem Grünau host without changing
# or duplicating the persistent three-worker matrix.
set -uo pipefail

GPU=${1:?physical GPU id}
shift
(( $# > 0 )) || { echo "usage: $0 GPU CONDITION:SEED ..."; exit 2; }
PY=${PY:-.venv/bin/python}
EVAL_PY=${EVAL_PY:-.venv2/bin/python}
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p runs/paper_causal_logs

for cell in "$@"; do
  condition=${cell%:*}
  seed=${cell##*:}
  run_name="${condition}_s${seed}"
  run="runs/$run_name"
  log="runs/paper_causal_logs/${run_name}.log"
  [[ -f "$run/EVAL_DONE" ]] && continue
  if [[ ! -f "$run/TRAIN_DONE" ]]; then
    if $PY scripts/train.py +experiment="$condition" \
        run_name="$run_name" seed="$seed" device=cuda:0 >>"$log" 2>&1; then
      touch "$run/TRAIN_DONE"
    else
      touch "$run/TRAIN_FAILED"
      continue
    fi
  fi
  if PY=$EVAL_PY bash scripts/eval_run.sh "$run" cuda:0 >>"$log" 2>&1; then
    rm -f "$run/EVAL_FAILED"
    touch "$run/EVAL_DONE"
  else
    touch "$run/EVAL_FAILED"
  fi
done

