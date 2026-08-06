#!/usr/bin/env bash
# Sparse multi-depth endpoint Energy factorial.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
py=${1:?python executable}
loss_mode=${2:?mse|rank|both}
dynamics_mode=${3:?energy_only|dense_dynamics}
geometry_mode=${4:?none|straight|projected_straight|monotone|projected_monotone}
lr=${5:-3e-4}
seed=${SEED:-0}; device=${DEVICE:-cuda:0}
epochs=${EPOCHS:-10}; train_size=${TRAIN_SIZE:-30000}
batch=${BATCH_SIZE:-16}; episodes=${N_EPISODES:-300}; width=${BEAM_WIDTH:-8}

rank_weight=0; mse_weight=0
case "$loss_mode" in
  mse) mse_weight=0.25 ;;
  rank) rank_weight=1 ;;
  both) rank_weight=1; mse_weight=0.25 ;;
  *) echo "unknown loss mode: $loss_mode" >&2; exit 2 ;;
esac
dense_weight=0
case "$dynamics_mode" in
  energy_only) ;;
  dense_dynamics) dense_weight=1 ;;
  *) echo "unknown dynamics mode: $dynamics_mode" >&2; exit 2 ;;
esac
straight_weight=0; monotone_weight=0; projected=false
case "$geometry_mode" in
  none) ;;
  straight) straight_weight=0.02 ;;
  projected_straight) straight_weight=0.02; projected=true ;;
  monotone) monotone_weight=0.1 ;;
  projected_monotone) monotone_weight=0.1; projected=true ;;
  *) echo "unknown geometry mode: $geometry_mode" >&2; exit 2 ;;
esac

tmp=${SLURM_TMPDIR:-/tmp/tj-multidepth-${SLURM_JOB_ID:-$$}}
mkdir -p "$tmp"; chmod 700 "$tmp"
export TMPDIR="$tmp" TMP="$tmp" TEMP="$tmp"
model_dir="$RUN_DIR/model"

"$py" "$TEXTJEPA_ROOT/scripts/train.py" \
  +experiment=paper_gar_scoring_screen \
  seed="$seed" device="$device" allow_legacy_predictor=true \
  train.lr="$lr" train.epochs="$epochs" train.batch_size="$batch" \
  train.num_workers=2 data.train_size="$train_size" data.val_size=500 \
  data.test_size=500 train.eval_batches=40 train.warmup_steps=500 \
  data.geo_rank_horizon=16 'data.geo_rank_horizons=null' \
  data.geo_rank_policy=random data.geo_rank_rollouts=2 \
  data.geo_rank_rollout_for_h1=true data.geo_rank_candidate_interface=feasible_menu \
  data.geo_rank_k=2 \
  model.geo_rank_score_mode=horizon model.geo_energy_target=distance \
  model.geo_horizon_supervise_prefixes=true \
  'model.geo_horizon_prefix_depths=[0,1,2,4,8,16]' \
  model.geo_proj="$projected" model.dense_rollout_depth=16 \
  objective.geo_rank.weight=0 objective.geo_energy_mse.weight=0 \
  objective.geo_advantage_mse.weight=0 \
  objective.geo_horizon_rank.weight="$rank_weight" \
  objective.geo_horizon_rank.kind=logistic \
  objective.geo_horizon_energy_mse.weight="$mse_weight" \
  objective.geo_horizon_straighten.weight="$straight_weight" \
  objective.geo_horizon_straighten.projected="$projected" \
  objective.geo_horizon_monotone.weight="$monotone_weight" \
  objective.geo_horizon_monotone.projected="$projected" \
  objective.counterfactual_state.weight=1 objective.latent_pred.weight=1 \
  objective.chunk_pred.weight=2 objective.vicreg.weight=1 \
  objective.dense_rollout.weight="$dense_weight" \
  hydra.run.dir="$model_dir" hydra.output_subdir=null

N_EPISODES="$episodes" BEAM_WIDTH="$width" EVAL_DEPTHS="1 2 4 8 16" \
  SEARCH_ALGORITHM=root_balanced_beam HYBRID_LOCAL_PRUNING=false \
  CANDIDATE_INTERFACE=feasible_menu \
  bash "$TEXTJEPA_ROOT/scripts/run_intent_terminal_energy_eval.sh" \
  "$py" "$model_dir/best.pt" \
  "multidepth-${loss_mode}-${dynamics_mode}-${geometry_mode}"

"$py" - "$RUN_DIR/training_complete.json" "$loss_mode" \
  "$dynamics_mode" "$geometry_mode" "$seed" "$lr" "$rank_weight" \
  "$mse_weight" "$dense_weight" "$straight_weight" "$monotone_weight" \
  "$projected" <<'PY'
import json, os, pathlib, sys
p = pathlib.Path(sys.argv[1]); q = p.with_suffix(".tmp")
q.write_text(json.dumps({
    "status": "completed", "loss_mode": sys.argv[2],
    "dynamics_mode": sys.argv[3], "geometry_mode": sys.argv[4],
    "seed": int(sys.argv[5]), "learning_rate": float(sys.argv[6]),
    "supervised_depths": [0, 1, 2, 4, 8, 16],
    "endpoint_rank_weight": float(sys.argv[7]),
    "absolute_energy_mse_weight": float(sys.argv[8]),
    "dense_dynamics_weight": float(sys.argv[9]),
    "straightening_weight": float(sys.argv[10]),
    "monotonicity_weight": float(sys.argv[11]),
    "projected_geometry": sys.argv[12].lower() == "true",
    "rollouts_per_root": 2, "root_distillation_weight": 0,
}, indent=2) + "\n")
os.replace(q, p)
PY
