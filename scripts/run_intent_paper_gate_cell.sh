#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
family=${2:?model family}
seed=${3:?seed}
width=${4:-128}
learning_rate=${5:-3e-4}
device=${DEVICE:-cuda:0}
model_dir="$RUN_DIR/model"

common=(
  "data=igsm_real"
  "seed=$seed"
  "device=$device"
  "train.lr=$learning_rate"
  "train.epochs=2"
  "train.batch_size=16"
  "train.num_workers=2"
  "train.warmup_steps=10"
  "data.train_size=256"
  "data.val_size=64"
  "data.test_size=64"
  "hydra.run.dir=$model_dir"
  "hydra.output_subdir=null"
)

case "$family" in
  geometry_jepa)
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/train.py" \
      +experiment=paper_causal_geometry_value "${common[@]}" \
      "model.d_model=$width" model.chunk_layers=1 model.chunk_heads=4 \
      model.state_layers=2 model.state_heads=4 model.predictor_layers=1 \
      model.predictor_heads=4 model.ff_mult=2 model.d_action=16 \
      model.macro_k=0 model.max_chunk_len=96 model.max_chunks=96
    eval_kind=jepa
    eval_extra=(--simulation-depth 2 --proposal-top-m 4 --beam-width 4)
    ;;
  looped_token_lm)
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/train_lm.py" \
      +experiment=paper_token_lm_looped "${common[@]}" \
      "model.d_model=$width" model.n_layers=2 model.n_heads=4 \
      model.ff_mult=2 model.max_len=1024
    eval_kind=token_lm
    eval_extra=(--eval-loops 4)
    ;;
  looped_sentence_lm|looped_sentence_latent_lm)
    experiment=paper_sentence_lm_looped
    score=decoder
    if [[ "$family" == "looped_sentence_latent_lm" ]]; then
      experiment=paper_sentence_latent_lm_looped
      score=latent
    fi
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/train_sentlm.py" \
      "+experiment=$experiment" "${common[@]}" \
      "model.d_model=$width" model.chunk_layers=1 model.chunk_heads=4 \
      model.state_layers=2 model.state_heads=4 model.dec_layers=1 \
      model.dec_heads=4 model.ff_mult=2 model.max_chunk_len=96 \
      model.max_chunks=96
    eval_kind=sentence_lm
    eval_extra=(--eval-loops 4 --sentence-score "$score")
    ;;
  *)
    echo "unknown gate family: $family" >&2
    exit 2
    ;;
esac

"$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_observed_action.py" \
  --kind "$eval_kind" --checkpoint "$model_dir/best.pt" \
  --device "$device" --split val --episodes 32 --excess-actions 0 2 \
  "${eval_extra[@]}" --out "$RUN_DIR/metrics.json"
