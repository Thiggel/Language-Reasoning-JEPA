#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
seed=${2:?seed}
d_model=${3:?model width}
chunk_layers=${4:?chunk encoder layers}
state_layers=${5:?state encoder layers}
learning_rate=${6:?learning rate}
epochs=${7:?epochs}
warmup_steps=${8:?warmup steps}
name="paper_causal_j3_d${d_model}_c${chunk_layers}_s${state_layers}_lr${learning_rate}_e${epochs}_seed${seed}"
model_dir="$RUN_DIR/model"

export TMPDIR="/tmp/tj-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/train.py" \
  "+experiment=paper_causal_j3" \
  "hydra.run.dir=$model_dir" \
  "run_name=$name" \
  "seed=$seed" \
  "model.d_model=$d_model" \
  "model.chunk_layers=$chunk_layers" \
  "model.state_layers=$state_layers" \
  "train.lr=$learning_rate" \
  "train.epochs=$epochs" \
  "train.warmup_steps=$warmup_steps" \
  "device=${DEVICE:-cuda:0}"

PY="$python_bin" bash "${TEXTJEPA_ROOT}/scripts/eval_run.sh" \
  "$model_dir" "${DEVICE:-cuda:0}"

jq -n \
  --argjson seed "$seed" \
  --argjson d_model "$d_model" \
  --argjson chunk_layers "$chunk_layers" \
  --argjson state_layers "$state_layers" \
  --arg learning_rate "$learning_rate" \
  --argjson epochs "$epochs" \
  --argjson warmup_steps "$warmup_steps" \
  --slurpfile strict "$model_dir/plan_slack0_look1.json" \
  --slurpfile slack "$model_dir/plan_slack2_look1.json" \
  '{seed: $seed, d_model: $d_model, chunk_layers: $chunk_layers,
    state_layers: $state_layers, learning_rate: $learning_rate,
    epochs: $epochs, warmup_steps: $warmup_steps,
    strict: $strict[0], slack2: $slack[0]}' \
  > "$RUN_DIR/metrics.json"
