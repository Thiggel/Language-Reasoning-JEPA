#!/usr/bin/env bash
# Faithful-iGSM confirmation. Do not launch before selecting the flat recipe.
set -uo pipefail
WORKER=${1:?worker id 0, 1, or 2}
GPU=${2:?physical GPU id}
PY=${PY:-.venv/bin/python}
EVAL_PY=${EVAL_PY:-.venv2/bin/python}
conditions=(
  paper_causal_faithful_j3
  paper_token_lm_faithful
  paper_sentence_lm_faithful
  paper_sentence_latent_lm_faithful
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
    if [[ "$condition" == paper_causal_faithful_j3 ]]; then
      train_script=scripts/train.py
    elif [[ "$condition" == paper_token_lm_faithful ]]; then
      train_script=scripts/train_lm.py
    else
      train_script=scripts/train_sentlm.py
    fi
    if ! $PY "$train_script" +experiment="$condition" \
      run_name="$run_name" seed="$seed" device=cuda:0; then
      touch "runs/${run_name}/TRAIN_FAILED"
      continue
    fi
    touch "runs/${run_name}/TRAIN_DONE"
    failed=0
    if [[ "$condition" == paper_causal_faithful_j3 ]]; then
      PY=$EVAL_PY bash scripts/eval_run.sh "runs/${run_name}" cuda:0 \
        || failed=1
    else
      for slack in 0 2; do
        if [[ "$condition" == paper_token_lm_faithful ]]; then
          $PY scripts/plan_lm.py ckpt="runs/${run_name}/best.pt" \
            device=cuda:0 slack=$slack || failed=1
        else
          $PY scripts/plan_sentlm.py ckpt="runs/${run_name}/best.pt" \
            device=cuda:0 slack=$slack score=decoder || failed=1
          if [[ "$condition" == paper_sentence_latent_lm_faithful ]]; then
            $PY scripts/plan_sentlm.py ckpt="runs/${run_name}/best.pt" \
              device=cuda:0 slack=$slack score=latent || failed=1
          fi
        fi
      done
    fi
    if [[ $failed == 0 ]]; then
      touch "runs/${run_name}/EVAL_DONE"
    else
      touch "runs/${run_name}/EVAL_FAILED"
    fi
  } >>"$log" 2>&1
done
