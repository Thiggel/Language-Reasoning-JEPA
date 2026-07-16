#!/usr/bin/env bash
# Three workers run this script with worker ids 0,1,2. For each condition the
# seed-to-GPU assignment rotates, preventing model seed from being confounded
# with the two V100s and one RTX 6000 available on this host.
set -uo pipefail

WORKER=${1:?worker id 0, 1, or 2}
GPU=${2:?physical GPU id}
PY=${PY:-.venv/bin/python}
EVAL_PY=${EVAL_PY:-.venv2/bin/python}

conditions=(
  paper_causal_j0
  paper_causal_j1
  paper_causal_j2
  paper_causal_j3
  paper_causal_j4_dense4
  paper_causal_a_nolatent
  paper_causal_a_nooutcome
  paper_causal_a_nooutroll
  paper_causal_a_novic
  paper_causal_a_noema
  paper_causal_a_residual
  paper_causal_a_ldad
  paper_causal_a_value
  paper_causal_a_monotone
  paper_causal_a_cfout
)

mkdir -p runs/paper_causal_logs
export CUDA_VISIBLE_DEVICES=$GPU

for i in "${!conditions[@]}"; do
  condition=${conditions[$i]}
  seed=$(( (i + WORKER) % 3 ))
  run_name="${condition}_s${seed}"
  log="runs/paper_causal_logs/${run_name}.log"
  if [[ -f "runs/${run_name}/EVAL_DONE" ]]; then
    continue
  fi
  {
    echo "START condition=${condition} seed=${seed} gpu=${GPU}"
    if [[ ! -f "runs/${run_name}/TRAIN_DONE" ]]; then
      if $PY scripts/train.py \
        +experiment="$condition" \
        run_name="$run_name" seed="$seed" device=cuda:0; then
        touch "runs/${run_name}/TRAIN_DONE"
      else
        touch "runs/${run_name}/TRAIN_FAILED"
        echo "TRAIN_FAILED ${run_name}"
        continue
      fi
    fi
    if PY=$EVAL_PY bash scripts/eval_run.sh "runs/${run_name}" cuda:0; then
      touch "runs/${run_name}/EVAL_DONE"
      echo "DONE ${run_name}"
    else
      touch "runs/${run_name}/EVAL_FAILED"
      echo "EVAL_FAILED ${run_name}"
    fi
  } >>"$log" 2>&1
done

# Retry evaluations that failed because of a transient evaluator/software
# issue, without retraining their completed checkpoints.
for i in "${!conditions[@]}"; do
  condition=${conditions[$i]}
  seed=$(( (i + WORKER) % 3 ))
  run_name="${condition}_s${seed}"
  log="runs/paper_causal_logs/${run_name}.log"
  [[ -f "runs/${run_name}/TRAIN_DONE" ]] || continue
  [[ -f "runs/${run_name}/EVAL_DONE" ]] && continue
  if PY=$EVAL_PY bash scripts/eval_run.sh "runs/${run_name}" cuda:0 >>"$log" 2>&1; then
    rm -f "runs/${run_name}/EVAL_FAILED"
    touch "runs/${run_name}/EVAL_DONE"
    echo "REPAIRED_EVAL ${run_name}" >>"$log"
  else
    touch "runs/${run_name}/EVAL_FAILED"
  fi
done
