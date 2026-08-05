#!/usr/bin/env bash
# Train one state-Energy rollout-exposure gate, then evaluate terminal beams.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
py=${1:?python executable}; variant=${2:?variant}; lr=${3:?learning rate}
seed=${SEED:-0}; device=${DEVICE:-cuda:0}
epochs=${EPOCHS:-10}; train_size=${TRAIN_SIZE:-30000}; batch=${BATCH_SIZE:-32}
rollout_depths='[]'; rollout_detach=false; rollout_mse=0; dense_depth=0; dense_weight=0
case "$variant" in
  base) ;;
  rollout)
    rollout_depths='[0,1,2,4]'; rollout_mse=0.25; dense_depth=4 ;;
  rollout_detached)
    rollout_depths='[0,1,2,4]'; rollout_detach=true
    rollout_mse=0.25; dense_depth=4 ;;
  rollout_dense)
    rollout_depths='[0,1,2,4]'; rollout_mse=0.25
    dense_depth=4; dense_weight=1.0 ;;
  *) echo "unknown state-Energy variant: $variant" >&2; exit 2 ;;
esac

tmp=${SLURM_TMPDIR:-/tmp/tj-state-energy-${SLURM_JOB_ID:-$$}}
mkdir -p "$tmp"; chmod 700 "$tmp"
export TMPDIR="$tmp" TMP="$tmp" TEMP="$tmp"
model_dir="$RUN_DIR/model"
"$py" "$TEXTJEPA_ROOT/scripts/train.py" \
  +experiment=paper_state_energy_rollout \
  seed="$seed" device="$device" train.lr="$lr" train.epochs="$epochs" \
  train.batch_size="$batch" train.num_workers=2 data.train_size="$train_size" \
  data.val_size=500 data.test_size=500 train.warmup_steps=500 \
  model.d_model=256 model.chunk_layers=2 model.chunk_heads=4 \
  model.state_layers=4 model.state_heads=8 model.predictor_layers=2 \
  model.ff_mult=4 model.d_action=16 model.max_chunk_len=96 model.max_chunks=96 \
  "model.geo_energy_rollout_depths=$rollout_depths" \
  model.geo_energy_rollout_detach_body="$rollout_detach" \
  model.dense_rollout_depth="$dense_depth" \
  objective.geo_rollout_energy_mse.weight="$rollout_mse" \
  objective.dense_rollout.weight="$dense_weight" \
  hydra.run.dir="$model_dir" hydra.output_subdir=null

"$py" - "$RUN_DIR/training_complete.json" "$variant" "$seed" "$lr" \
  "$rollout_depths" "$rollout_detach" "$rollout_mse" \
  "$dense_depth" "$dense_weight" <<'PY'
import json, os, pathlib, sys
p=pathlib.Path(sys.argv[1]); q=p.with_suffix('.tmp')
q.write_text(json.dumps({
    'status':'completed', 'variant':sys.argv[2], 'seed':int(sys.argv[3]),
    'learning_rate':float(sys.argv[4]),
    'rollout_depths':json.loads(sys.argv[5]),
    'rollout_detach_body':sys.argv[6].lower() == 'true',
    'rollout_energy_mse_weight':float(sys.argv[7]),
    'dense_rollout_depth':int(sys.argv[8]),
    'dense_rollout_weight':float(sys.argv[9]),
}, indent=2)+'\n')
os.replace(q,p)
PY

N_EPISODES="${N_EPISODES:-200}" BEAM_WIDTH="${BEAM_WIDTH:-8}" \
  bash "$TEXTJEPA_ROOT/scripts/run_intent_terminal_energy_eval.sh" \
  "$py" "$model_dir/best.pt" "$variant"
