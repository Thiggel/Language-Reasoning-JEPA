#!/usr/bin/env bash
# One direct-ranker, ranking-loss, or horizon-conditioned GAR screen cell.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
py=${1:?python executable}; family=${2:?family}; first=${3:?first axis}
second=${4:?second axis}; lr=${5:-3e-4}; seed=${SEED:-0}
device=${DEVICE:-cuda:0}; epochs=${EPOCHS:-10}
train_size=${TRAIN_SIZE:-30000}; batch=${BATCH_SIZE:-32}
val_size=${VAL_SIZE:-500}; eval_batches=${EVAL_BATCHES:-40}

score_mode=value; energy_target=distance; rank_kind=hinge
rank=0; direct_mse=0; horizon_rank=0; depths='[]'
rollout_rank=0; rollout_direct=0; dense_depth=0; dense_weight=0
composition=terminal; search_algorithm=beam; horizon=1; rollouts=1

case "$family" in
  direct)
    score_mode=direct; energy_target=advantage; direct_mse=0.25
    case "$first" in mse) ;; mse_rank) rank=1 ;; *) exit 2 ;; esac
    case "$second" in
      one_step) ;;
      dense4)
        depths='[1,2,3,4]'; dense_depth=4; dense_weight=1
        rollout_direct=0.25
        [[ "$first" == mse_rank ]] && rollout_rank=1
        ;;
      *) exit 2 ;;
    esac
    ;;
  rankloss)
    rank_kind=$first; rank=1; direct_mse=0.25
    depths='[1,2,3,4]'; dense_depth=4; dense_weight=1
    rollout_rank=1; rollout_direct=0.25
    case "$second" in
      state) score_mode=value; energy_target=distance; composition=terminal ;;
      difference)
        score_mode=transition; energy_target=advantage
        composition=cumulative
        ;;
      *) exit 2 ;;
    esac
    ;;
  horizon)
    rank_kind=$first; score_mode=horizon; horizon_rank=1
    dense_depth=4; dense_weight=1; horizon=4; rollouts=4
    search_algorithm=root_balanced_beam
    ;;
  *) echo "unknown family: $family" >&2; exit 2 ;;
esac

model_dir="$RUN_DIR/model"
tmp=${SLURM_TMPDIR:-/tmp/tj-gar-scoring-${SLURM_JOB_ID:-$$}}
mkdir -p "$tmp"; chmod 700 "$tmp"
export TMPDIR="$tmp" TMP="$tmp" TEMP="$tmp"
"$py" "$TEXTJEPA_ROOT/scripts/train.py" \
  +experiment=paper_gar_scoring_screen \
  seed="$seed" device="$device" train.lr="$lr" train.epochs="$epochs" \
  train.batch_size="$batch" train.num_workers=2 data.train_size="$train_size" \
  data.val_size="$val_size" data.test_size="$val_size" \
  train.eval_batches="$eval_batches" train.warmup_steps=500 \
  data.geo_rank_horizon="$horizon" data.geo_rank_policy=random \
  data.geo_rank_rollouts="$rollouts" \
  model.geo_rank_score_mode="$score_mode" \
  model.geo_energy_target="$energy_target" \
  "model.geo_rank_rollout_depths=$depths" \
  model.dense_rollout_depth="$dense_depth" \
  objective.geo_rank.weight="$rank" \
  objective.geo_rank.kind="$rank_kind" \
  objective.geo_advantage_mse.weight=0 \
  objective.geo_energy_mse.weight="$direct_mse" \
  objective.geo_rollout_rank.weight="$rollout_rank" \
  objective.geo_rollout_rank.kind="$rank_kind" \
  objective.geo_rollout_advantage_mse.weight=0 \
  objective.geo_rollout_candidate_mse.weight="$rollout_direct" \
  objective.geo_horizon_rank.weight="$horizon_rank" \
  objective.geo_horizon_rank.kind="$rank_kind" \
  objective.dense_rollout.weight="$dense_weight" \
  hydra.run.dir="$model_dir" hydra.output_subdir=null

N_EPISODES="${N_EPISODES:-300}" BEAM_WIDTH="${BEAM_WIDTH:-8}" \
TRANSITION_ENERGY_COMPOSITION="$composition" \
SEARCH_ALGORITHM="$search_algorithm" \
  bash "$TEXTJEPA_ROOT/scripts/run_intent_terminal_energy_eval.sh" \
  "$py" "$model_dir/best.pt" "${family}-${first}-${second}"
