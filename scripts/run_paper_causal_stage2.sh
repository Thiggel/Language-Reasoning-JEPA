#!/usr/bin/env bash
# Selection curves; invoke three workers exactly as for Stage 1.
set -uo pipefail
WORKER=${1:?worker id 0, 1, or 2}
GPU=${2:?physical GPU id}
PY=${PY:-.venv/bin/python}
EVAL_PY=${EVAL_PY:-.venv2/bin/python}
conditions=(
  paper_causal_symbolic_reference
  paper_causal_gar_h1
  paper_causal_gar_h4
  paper_causal_gar_h8
  paper_causal_gar_h16
  paper_causal_gar_k1
  paper_causal_gar_k4
  paper_causal_gar_k8
  paper_causal_dense1
  paper_causal_dense2
  paper_causal_dense8
  paper_causal_dense4_l05
  paper_causal_dense4_l07
)
mkdir -p runs/paper_causal_logs
export CUDA_VISIBLE_DEVICES=$GPU
for i in "${!conditions[@]}"; do
  condition=${conditions[$i]}
  seed=$(( (i + WORKER) % 3 ))
  run_name="${condition}_s${seed}"
  log="runs/paper_causal_logs/${run_name}.log"
  [[ -f "runs/${run_name}/EVAL_DONE" ]] && continue
  {
    echo "START condition=${condition} seed=${seed} gpu=${GPU}"
    if $PY scripts/train.py +experiment="$condition" \
      run_name="$run_name" seed="$seed" device=cuda:0; then
      touch "runs/${run_name}/TRAIN_DONE"
    else
      touch "runs/${run_name}/TRAIN_FAILED"
      continue
    fi
    if PY=$EVAL_PY bash scripts/eval_run.sh "runs/${run_name}" cuda:0; then
      touch "runs/${run_name}/EVAL_DONE"
    else
      touch "runs/${run_name}/EVAL_FAILED"
    fi
  } >>"$log" 2>&1
done
