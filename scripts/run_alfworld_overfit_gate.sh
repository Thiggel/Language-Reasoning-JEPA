#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" || -z "${TEXTJEPA_ROOT:-}" ]]; then
  echo "RUN_DIR and TEXTJEPA_ROOT must be supplied by researchctl" >&2
  exit 2
fi
python_bin=${1:?training python}
seed=${2:?seed}
learning_rate=${3:?learning rate}
epochs=${4:-300}
device=${DEVICE:-cuda:0}
pilot_root=${ALFWORLD_PILOT_ROOT:-/vol/home-vol2/ml/laitenbf/TextJEPA/data/intent_phrase/alfworld/pilot_v1}
alfworld_data=${ALFWORLD_DATA:-/vol/home-vol2/ml/laitenbf/TextJEPA/data/intent_phrase/alfworld_engine}

job_token=${SLURM_JOB_ID:-$$}
worker_tmp=/tmp/awo-$job_token
mkdir -p "$worker_tmp" && chmod 700 "$worker_tmp"
trap 'rm -rf "$worker_tmp"' EXIT
export TMPDIR=$worker_tmp TMP=$worker_tmp TEMP=$worker_tmp
export ALFWORLD_PILOT_ROOT=$pilot_root ALFWORLD_DATA=$alfworld_data

model_dir=$RUN_DIR/model
"$python_bin" "$TEXTJEPA_ROOT/scripts/train.py" \
  +experiment=paper_causal_geometry_value data=alfworld_pilot \
  seed="$seed" device="$device" train.lr="$learning_rate" \
  train.epochs="$epochs" train.batch_size=2 train.num_workers=0 \
  train.warmup_steps=20 train.eval_batches=2 train.log_every=10 \
  data.geo_rank_k=1 data.geo_rank_horizon=4 \
  model.d_model=64 model.chunk_layers=1 model.chunk_heads=4 \
  model.state_layers=2 model.state_heads=4 model.predictor_layers=2 \
  model.predictor_heads=4 model.ff_mult=2 model.d_action=16 \
  model.max_chunk_len=192 model.max_chunks=96 \
  hydra.run.dir="$model_dir" hydra.output_subdir=null

if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  exit 0
fi

eval_python=${ALFWORLD_PYTHON:-/vol/home-vol2/ml/laitenbf/.venv-alfworld/bin/python}
project_site=${TEXTJEPA_PROJECT_SITE:-/vol/home-vol2/ml/laitenbf/TextJEPA/.venv/lib64/python3.11/site-packages}
export PYTHONPATH=$TEXTJEPA_ROOT/src:$project_site
checkpoint=$model_dir/last.pt
"$eval_python" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
  --kind jepa --checkpoint "$checkpoint" --device "$device" --split train \
  --episodes 8 --excess-actions 0 4 --simulation-depth 2 \
  --proposal-top-m 8 --beam-width 4 --out "$RUN_DIR/train_metrics.json"
"$eval_python" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
  --kind jepa --checkpoint "$checkpoint" --device "$device" --split val \
  --episodes 4 --excess-actions 0 4 --simulation-depth 2 \
  --proposal-top-m 8 --beam-width 4 --out "$RUN_DIR/metrics.json"
