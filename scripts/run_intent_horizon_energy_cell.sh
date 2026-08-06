#!/usr/bin/env bash
# Horizon-conditioned endpoint-Energy training and hybrid-planning cells.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
py=${1:?python executable}; variant=${2:?variant}; lr=${3:-3e-4}
seed=${SEED:-0}; device=${DEVICE:-cuda:0}; epochs=${EPOCHS:-10}
train_size=${TRAIN_SIZE:-30000}; batch=${BATCH_SIZE:-32}
episodes=${N_EPISODES:-300}; width=${BEAM_WIDTH:-8}

horizon=4; horizons=null; dense_depth=4; dense_weight=1
dense_discount=1; rollouts=4; frozen=false
root_distill_weight=0.25; candidate_interface=feasible_menu; rank_k=2
feasible_k=null; invalid_k=null; invalid_mode=noop; prefix_energy=false
case "$variant" in
  fixed_h4) ;;
  mix_uniform)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=8 ;;
  mix_discount07)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=8; dense_discount=0.7 ;;
  mix_discount05)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=8; dense_discount=0.5 ;;
  mix_dense025)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=8; dense_weight=0.25 ;;
  mix_no_dense)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0 ;;
  mix_no_dense_aux0)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0
    root_distill_weight=0 ;;
  mix_no_dense_aux05)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0
    root_distill_weight=0.5 ;;
  mix_no_dense_aux1)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0
    root_distill_weight=1 ;;
  fixed_h4_distill025)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0
    root_distill_weight=0.25 ;;
  fixed_h4_distill1)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0
    root_distill_weight=1 ;;
  mix_no_dense_full_catalogue)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0
    candidate_interface=full_catalogue; rank_k=-1 ;;
  full_catalogue_noop_balanced)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0
    candidate_interface=full_catalogue; rank_k=-1
    feasible_k=2; invalid_k=2 ;;
  full_catalogue_failure_all)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0
    candidate_interface=full_catalogue; rank_k=-1; invalid_mode=failure ;;
  full_catalogue_failure_balanced)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0
    candidate_interface=full_catalogue; rank_k=-1; invalid_mode=failure
    feasible_k=2; invalid_k=2 ;;
  dense_endpoint_h2)
    horizon=2; horizons=null; dense_depth=0; dense_weight=0
    prefix_energy=true ;;
  dense_endpoint_h4)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0
    prefix_energy=true ;;
  dense_endpoint_h8)
    horizon=8; horizons=null; dense_depth=0; dense_weight=0
    prefix_energy=true ;;
  dense_endpoint_h16)
    horizon=16; horizons=null; dense_depth=0; dense_weight=0
    prefix_energy=true ;;
  frozen_mix)
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0
    frozen=true ;;
  *) echo "unknown horizon-Energy variant: $variant" >&2; exit 2 ;;
esac

model_dir="$RUN_DIR/model"
tmp=${SLURM_TMPDIR:-/tmp/tj-horizon-energy-${SLURM_JOB_ID:-$$}}
mkdir -p "$tmp"; chmod 700 "$tmp"
export TMPDIR="$tmp" TMP="$tmp" TEMP="$tmp"

extra=()
if [[ "$frozen" == true ]]; then
  init_ckpt=${INIT_CKPT:?frozen_mix requires INIT_CKPT}
  extra+=(
    "train.init_ckpt=$init_ckpt" train.init_mode=full
    train.reset_horizon_energy_head=true train.freeze_low_level=true
    train.train_high_level=false train.train_horizon_energy_head=true
  )
fi

"$py" "$TEXTJEPA_ROOT/scripts/train.py" \
  +experiment=paper_gar_scoring_screen \
  seed="$seed" device="$device" allow_legacy_predictor=true \
  train.lr="$lr" train.epochs="$epochs" train.batch_size="$batch" \
  train.num_workers=2 data.train_size="$train_size" data.val_size=500 \
  data.test_size=500 train.eval_batches=40 train.warmup_steps=500 \
  data.geo_rank_horizon="$horizon" "data.geo_rank_horizons=$horizons" \
  data.geo_rank_policy=random data.geo_rank_rollouts="$rollouts" \
  data.geo_rank_rollout_for_h1=true \
  data.geo_rank_candidate_interface="$candidate_interface" \
  data.geo_rank_k="$rank_k" \
  data.geo_rank_feasible_k="$feasible_k" \
  data.geo_rank_invalid_k="$invalid_k" \
  data.invalid_action_mode="$invalid_mode" \
  model.geo_rank_score_mode=horizon model.geo_energy_target=distance \
  model.geo_horizon_supervise_prefixes="$prefix_energy" \
  model.dense_rollout_depth="$dense_depth" \
  objective.geo_rank.weight=0 objective.geo_energy_mse.weight=0 \
  objective.geo_advantage_mse.weight="$root_distill_weight" \
  objective.geo_horizon_rank.weight=1 \
  objective.geo_horizon_rank.kind=logistic \
  objective.dense_rollout.weight="$dense_weight" \
  objective.dense_rollout.horizon_discount="$dense_discount" \
  "${extra[@]}" hydra.run.dir="$model_dir" hydra.output_subdir=null

if [[ "$frozen" == true ]]; then
  mkdir -p "$RUN_DIR/horizon_only"
  RUN_DIR="$RUN_DIR/horizon_only" N_EPISODES="$episodes" BEAM_WIDTH="$width" \
    SEARCH_ALGORITHM=root_balanced_beam HYBRID_LOCAL_PRUNING=false \
    CANDIDATE_INTERFACE="$candidate_interface" \
    INVALID_ACTION_MODE="$invalid_mode" \
    bash "$TEXTJEPA_ROOT/scripts/run_intent_terminal_energy_eval.sh" \
    "$py" "$model_dir/best.pt" "$variant-horizon-only"
  N_EPISODES="$episodes" BEAM_WIDTH="$width" \
    SEARCH_ALGORITHM=root_balanced_beam HYBRID_LOCAL_PRUNING=true \
    CANDIDATE_INTERFACE="$candidate_interface" \
    INVALID_ACTION_MODE="$invalid_mode" \
    bash "$TEXTJEPA_ROOT/scripts/run_intent_terminal_energy_eval.sh" \
    "$py" "$model_dir/best.pt" "$variant-hybrid"
else
  N_EPISODES="$episodes" BEAM_WIDTH="$width" \
    SEARCH_ALGORITHM=root_balanced_beam HYBRID_LOCAL_PRUNING=false \
    CANDIDATE_INTERFACE="$candidate_interface" \
    INVALID_ACTION_MODE="$invalid_mode" \
    bash "$TEXTJEPA_ROOT/scripts/run_intent_terminal_energy_eval.sh" \
    "$py" "$model_dir/best.pt" "$variant"
fi

"$py" - "$RUN_DIR" "$variant" "$seed" "$lr" "$horizons" \
  "$dense_depth" "$dense_weight" "$dense_discount" "$frozen" \
  "$root_distill_weight" "$candidate_interface" "$rank_k" \
  "$feasible_k" "$invalid_k" "$invalid_mode" "$prefix_energy" <<'PY'
import json, pathlib, sys
r = pathlib.Path(sys.argv[1])
(r / "training_complete.json").write_text(json.dumps({
    "status": "completed", "variant": sys.argv[2],
    "seed": int(sys.argv[3]), "learning_rate": float(sys.argv[4]),
    "training_horizons": sys.argv[5],
    "dense_rollout_depth": int(sys.argv[6]),
    "dense_rollout_weight": float(sys.argv[7]),
    "dense_rollout_discount": float(sys.argv[8]),
    "frozen_body": sys.argv[9].lower() == "true",
    "root_distill_weight": float(sys.argv[10]),
    "candidate_interface": sys.argv[11],
    "rank_alternatives": int(sys.argv[12]),
    "feasible_alternatives": sys.argv[13],
    "invalid_alternatives": sys.argv[14],
    "invalid_action_mode": sys.argv[15],
    "supervise_energy_prefixes": sys.argv[16].lower() == "true",
}, indent=2) + "\n")
PY
