#!/usr/bin/env bash
# Complete target-network × regularizer × faithful-LDAD factorial around J3.
set -uo pipefail
WORKER=${1:?worker id 0, 1, or 2}
GPU=${2:?physical GPU id}
PY=${PY:-.venv/bin/python}
EVAL_PY=${EVAL_PY:-.venv2/bin/python}
targets=(ema sg online)
regularizers=(none vicreg sigreg)
ldad_values=(off on)
conditions=()
for target in "${targets[@]}"; do
  for regularizer in "${regularizers[@]}"; do
    for ldad in "${ldad_values[@]}"; do
      conditions+=("${target}_${regularizer}_${ldad}")
    done
  done
done
mkdir -p runs/paper_causal_logs
export CUDA_VISIBLE_DEVICES=$GPU
for i in "${!conditions[@]}"; do
  cell=${conditions[$i]}
  # Exact cells already present in Stage 1: J3, LDAD, no-VICReg, and no-EMA.
  case "$cell" in
    ema_vicreg_off|ema_vicreg_on|ema_none_off|sg_vicreg_off) continue ;;
  esac
  IFS=_ read -r target regularizer ldad <<<"$cell"
  target_mode=$target
  [[ "$target" == sg ]] && target_mode=online
  [[ "$target" == online ]] && target_mode=online_nosg
  seed=$(( (i + WORKER) % 3 ))
  run_name="paper_causal_factorial_${cell}_s${seed}"
  log="runs/paper_causal_logs/${run_name}.log"
  [[ -f "runs/${run_name}/EVAL_DONE" ]] && continue
  vic=0.0
  sig=0.0
  ldad_weight=0.0
  ldad_enabled=false
  [[ "$regularizer" == vicreg ]] && vic=1.0
  [[ "$regularizer" == sigreg ]] && sig=0.01
  if [[ "$ldad" == on ]]; then
    ldad_weight=0.2
    ldad_enabled=true
  fi
  {
    if ! $PY scripts/train.py +experiment=paper_causal_j3 \
      run_name="$run_name" seed="$seed" device=cuda:0 \
      model.state_target="$target_mode" \
      model.observed_action_ldad="$ldad_enabled" \
      objective.vicreg.weight="$vic" \
      objective.sigreg.weight="$sig" \
      objective.observed_action_ldad.weight="$ldad_weight"; then
      touch "runs/${run_name}/TRAIN_FAILED"
      continue
    fi
    touch "runs/${run_name}/TRAIN_DONE"
    if PY=$EVAL_PY bash scripts/eval_run.sh "runs/${run_name}" cuda:0; then
      touch "runs/${run_name}/EVAL_DONE"
    else
      touch "runs/${run_name}/EVAL_FAILED"
    fi
  } >>"$log" 2>&1
done
