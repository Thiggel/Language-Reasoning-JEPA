#!/usr/bin/env bash
# One matched GAR calibration x rollout-exposure cell plus depth evaluation.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
py=${1:?python executable}; calibration=${2:?calibration}; rollout=${3:?rollout}
lr=${4:?learning rate}; seed=${SEED:-0}; device=${DEVICE:-cuda:0}
epochs=${EPOCHS:-10}; train_size=${TRAIN_SIZE:-30000}; batch=${BATCH_SIZE:-32}

score_mode=value; energy_target=advantage; horizon=2
pair_mse=0; direct_mse=0; composition=terminal
case "$calibration" in
  ranking) ;;
  pairwise) pair_mse=0.25 ;;
  advantage)
    score_mode=transition; energy_target=advantage; horizon=1
    direct_mse=0.25; composition=cumulative ;;
  state)
    score_mode=value; energy_target=distance; horizon=1
    direct_mse=0.25 ;;
  *) echo "unknown calibration: $calibration" >&2; exit 2 ;;
esac

depths='[]'; detach=false; rollout_rank=0; rollout_pair=0
rollout_direct=0; dense_depth=0; dense_weight=0
case "$rollout" in
  base) ;;
  joint|detached|dense)
    depths='[1,2,4]'; dense_depth=4; rollout_rank=1
    [[ "$rollout" == detached ]] && detach=true
    [[ "$rollout" == dense ]] && dense_weight=1
    if [[ "$calibration" == pairwise ]]; then
      rollout_pair=0.25
    elif [[ "$calibration" == advantage || "$calibration" == state ]]; then
      rollout_direct=0.25
    fi ;;
  *) echo "unknown rollout mode: $rollout" >&2; exit 2 ;;
esac

tmp=${SLURM_TMPDIR:-/tmp/tj-gar-factorial-${SLURM_JOB_ID:-$$}}
mkdir -p "$tmp"; chmod 700 "$tmp"
export TMPDIR="$tmp" TMP="$tmp" TEMP="$tmp"
model_dir="$RUN_DIR/model"
"$py" "$TEXTJEPA_ROOT/scripts/train.py" \
  +experiment=paper_gar_rollout_factorial \
  seed="$seed" device="$device" train.lr="$lr" train.epochs="$epochs" \
  train.batch_size="$batch" train.num_workers=2 data.train_size="$train_size" \
  data.val_size=500 data.test_size=500 train.warmup_steps=500 \
  data.geo_rank_horizon="$horizon" \
  model.geo_rank_score_mode="$score_mode" \
  model.geo_energy_target="$energy_target" \
  "model.geo_rank_rollout_depths=$depths" \
  model.geo_rank_rollout_detach_body="$detach" \
  model.dense_rollout_depth="$dense_depth" \
  objective.geo_advantage_mse.weight="$pair_mse" \
  objective.geo_energy_mse.weight="$direct_mse" \
  objective.geo_rollout_rank.weight="$rollout_rank" \
  objective.geo_rollout_advantage_mse.weight="$rollout_pair" \
  objective.geo_rollout_candidate_mse.weight="$rollout_direct" \
  objective.dense_rollout.weight="$dense_weight" \
  hydra.run.dir="$model_dir" hydra.output_subdir=null

"$py" - "$RUN_DIR/training_complete.json" "$calibration" "$rollout" \
  "$seed" "$lr" "$composition" <<'PY'
import json, os, pathlib, sys
p=pathlib.Path(sys.argv[1]); q=p.with_suffix('.tmp')
q.write_text(json.dumps({
    'status':'completed', 'calibration':sys.argv[2], 'rollout':sys.argv[3],
    'seed':int(sys.argv[4]), 'learning_rate':float(sys.argv[5]),
    'transition_energy_composition':sys.argv[6],
}, indent=2)+'\n')
os.replace(q,p)
PY

N_EPISODES="${N_EPISODES:-300}" BEAM_WIDTH="${BEAM_WIDTH:-8}" \
TRANSITION_ENERGY_COMPOSITION="$composition" \
  bash "$TEXTJEPA_ROOT/scripts/run_intent_terminal_energy_eval.sh" \
  "$py" "$model_dir/best.pt" "${calibration}-${rollout}"
