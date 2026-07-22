#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" || -z "${TEXTJEPA_ROOT:-}" ]]; then
  echo "RUN_DIR and TEXTJEPA_ROOT must be supplied by researchctl" >&2
  exit 2
fi
python_bin=${1:?training python}
family=${2:?geometry_jepa, token_lm, sentence_lm, or sentence_latent_lm}
seed=${3:?seed}
learning_rate=${4:?learning rate}
epochs=${5:?epochs}
device=${DEVICE:-cuda:0}
shared_root=${TEXTJEPA_SHARED_ROOT:-/vol/home-vol2/ml/laitenbf/TextJEPA}
pilot_root=${ALFWORLD_PILOT_ROOT:-$shared_root/data/intent_phrase/alfworld/pilot_v2}
alfworld_data=${ALFWORLD_DATA:-$shared_root/data/intent_phrase/alfworld_engine}

job_token=${SLURM_JOB_ID:-$$}
worker_tmp=/tmp/awnp-$job_token
mkdir -p "$worker_tmp" && chmod 700 "$worker_tmp"
trap 'rm -rf "$worker_tmp"' EXIT
export TMPDIR=$worker_tmp TMP=$worker_tmp TEMP=$worker_tmp
export ALFWORLD_PILOT_ROOT=$pilot_root ALFWORLD_DATA=$alfworld_data

model_dir=$RUN_DIR/model
common=(
  data=alfworld_pilot seed="$seed" device="$device"
  train.lr="$learning_rate" train.epochs="$epochs"
  train.batch_size=2 train.num_workers=0 train.warmup_steps=40
  train.log_every=20 hydra.run.dir="$model_dir" hydra.output_subdir=null
)
case "$family" in
  geometry_jepa)
    "$python_bin" "$TEXTJEPA_ROOT/scripts/train.py" \
      +experiment=paper_causal_geometry_gar_no_prior "${common[@]}" \
      train.eval_batches=2 data.geo_rank_k=2 data.geo_rank_horizon=4 \
      data.dense_geo_anchors=true \
      model.d_model=64 model.chunk_layers=1 model.chunk_heads=4 \
      model.state_layers=2 model.state_heads=4 model.predictor_layers=2 \
      model.predictor_heads=4 model.ff_mult=2 model.d_action=16 \
      model.max_chunk_len=192 model.max_chunks=96
    eval_kind=jepa
    eval_extra=(--jepa-candidate-mode full --beam-width 4)
    ;;
  token_lm)
    "$python_bin" "$TEXTJEPA_ROOT/scripts/train_lm.py" \
      +experiment=paper_token_lm "${common[@]}" \
      model.d_model=64 model.n_layers=2 model.n_heads=4 \
      model.ff_mult=2 model.max_len=1536
    eval_kind=token_lm
    eval_extra=()
    ;;
  sentence_lm|sentence_latent_lm)
    experiment=paper_sentence_lm
    score=decoder
    if [[ "$family" == sentence_latent_lm ]]; then
      experiment=paper_sentence_latent_lm
      score=latent
    fi
    "$python_bin" "$TEXTJEPA_ROOT/scripts/train_sentlm.py" \
      "+experiment=$experiment" "${common[@]}" \
      model.d_model=64 model.chunk_layers=1 model.chunk_heads=4 \
      model.state_layers=2 model.state_heads=4 model.dec_layers=1 \
      model.dec_heads=4 model.ff_mult=2 model.max_chunk_len=192 \
      model.max_chunks=128
    eval_kind=sentence_lm
    eval_extra=(--sentence-score "$score")
    ;;
  *)
    echo "unknown model family: $family" >&2
    exit 2
    ;;
esac

if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  exit 0
fi

eval_python=${ALFWORLD_PYTHON:-/vol/home-vol2/ml/laitenbf/.venv-alfworld/bin/python}
project_site=${TEXTJEPA_PROJECT_SITE:-$shared_root/.venv/lib64/python3.11/site-packages}
export PYTHONPATH=$TEXTJEPA_ROOT/src:$project_site
checkpoint=$model_dir/last.pt
if [[ "$family" == geometry_jepa ]]; then
  "$eval_python" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
    --kind "$eval_kind" --checkpoint "$checkpoint" --device "$device" \
    --split train --episodes 8 --excess-actions 0 4 \
    --simulation-depth 1 "${eval_extra[@]}" \
    --out "$RUN_DIR/train_depth1_metrics.json"
  "$eval_python" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
    --kind "$eval_kind" --checkpoint "$checkpoint" --device "$device" \
    --split train --episodes 8 --excess-actions 0 4 \
    --simulation-depth 2 "${eval_extra[@]}" \
    --out "$RUN_DIR/train_metrics.json"
  "$eval_python" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
    --kind "$eval_kind" --checkpoint "$checkpoint" --device "$device" \
    --split val --episodes 4 --excess-actions 0 4 \
    --simulation-depth 2 "${eval_extra[@]}" \
    --out "$RUN_DIR/metrics.json"
else
  "$eval_python" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
    --kind "$eval_kind" --checkpoint "$checkpoint" --device "$device" \
    --split train --episodes 8 --excess-actions 0 4 "${eval_extra[@]}" \
    --out "$RUN_DIR/train_metrics.json"
  "$eval_python" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
    --kind "$eval_kind" --checkpoint "$checkpoint" --device "$device" \
    --split val --episodes 4 --excess-actions 0 4 "${eval_extra[@]}" \
    --out "$RUN_DIR/metrics.json"
fi
