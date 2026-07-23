#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" || -z "${TEXTJEPA_ROOT:-}" ]]; then
  echo "RUN_DIR and TEXTJEPA_ROOT must be supplied by researchctl" >&2
  exit 2
fi
python_bin=${1:?python executable}
seed=${2:-0}
device=${DEVICE:-cuda:0}
job_token=${SLURM_JOB_ID:-$$}
worker_tmp=${SLURM_TMPDIR:-/tmp/iipa-$job_token}
if (( ${#worker_tmp} > 60 )); then
  worker_tmp=/tmp/iipa-$job_token
fi
mkdir -p "$worker_tmp"
chmod 700 "$worker_tmp"
trap 'rm -rf "$worker_tmp"' EXIT
export TMPDIR=$worker_tmp TMP=$worker_tmp TEMP=$worker_tmp

for reference in random oracle; do
  "$python_bin" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
    --kind "$reference" \
    --data-config "$TEXTJEPA_ROOT/configs/data/igsm_real.yaml" \
    --device "$device" --split val --episodes 64 \
    --excess-actions 0 1 2 4 --seed 7321 \
    --out "$RUN_DIR/${reference}_metrics.json"
done

train_cell() {
  local name=$1 learning_rate=$2 shuffled=$3
  local model_dir=$RUN_DIR/$name/model
  "$python_bin" "$TEXTJEPA_ROOT/scripts/train.py" \
    +experiment=paper_causal_geometry_gar_no_prior data=igsm_real \
    data.train_size=64 data.val_size=32 data.test_size=32 \
    data.fresh_per_epoch=false "data.shuffle_actions=$shuffled" \
    data.geo_rank_k=4 data.geo_rank_horizon=2 \
    data.dense_geo_anchors=true "seed=$seed" "device=$device" \
    "train.lr=$learning_rate" train.epochs=20 train.batch_size=4 \
    train.num_workers=0 train.warmup_steps=20 train.eval_batches=2 \
    train.log_every=20 model.d_model=64 model.chunk_layers=1 \
    model.chunk_heads=4 model.state_layers=2 model.state_heads=4 \
    model.predictor_layers=2 model.predictor_heads=4 model.ff_mult=2 \
    model.d_action=16 model.max_chunk_len=96 model.max_chunks=96 \
    model.dropout=0.0 hydra.run.dir="$model_dir" \
    hydra.output_subdir=null
  for split in train val; do
    "$python_bin" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
      --kind jepa --checkpoint "$model_dir/last.pt" --device "$device" \
      --split "$split" --episodes 64 --excess-actions 0 1 2 4 \
      --simulation-depth 2 --jepa-candidate-mode full --beam-width 4 \
      --out "$RUN_DIR/${name}_${split}_metrics.json"
  done
}

train_cell geometry_lr1e3 0.001 false
train_cell geometry_lr3e3 0.003 false
train_cell shuffled_actions_lr1e3 0.001 true
"$python_bin" "$TEXTJEPA_ROOT/scripts/audit_intent_checkpoint_invariants.py" \
  --checkpoint "$RUN_DIR/geometry_lr1e3/model/last.pt" \
  --device cpu --out "$RUN_DIR/checkpoint_invariants.json"
