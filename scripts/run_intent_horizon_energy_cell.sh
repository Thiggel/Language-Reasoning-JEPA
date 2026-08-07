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
horizon_rank_weight=1; counterfactual_weight=1; latent_weight=1
ranking_kind=logistic; chunk_weight=2; vicreg_weight=1
horizon_input=true; predictor_residual=true; td_auxiliary=none
factual_only=false
score_mode=horizon; rollout_for_h1=true; td_q_weight=0; expectile_weight=0
td_jepa_weight=0; goal_head_weight=0
case "$variant" in
  fixed_h4) ;;
  mix_pow2_16_aux0)
    # Coherent supported Energy: every reported eval depth is a training
    # horizon; no root pair-difference auxiliary (single head semantics).
    horizon=16; horizons='[1,2,4,8,16]'; dense_depth=0; dense_weight=0
    root_distill_weight=0 ;;
  mix_pow2_16_aux0_nohorizon)
    # Horizon-blind control: one shared Energy for every depth.
    horizon=16; horizons='[1,2,4,8,16]'; dense_depth=0; dense_weight=0
    root_distill_weight=0; horizon_input=false ;;
  mix_pow2_16_aux025_nohorizon)
    # Horizon-blind + root pair-difference auxiliary: with one shared Energy
    # semantics the auxiliary is coherent (it distills multi-step quality
    # into the same Energy the planner queries at every depth).
    horizon=16; horizons='[1,2,4,8,16]'; dense_depth=0; dense_weight=0
    root_distill_weight=0.25; horizon_input=false ;;
  mix4_aux025_nohorizon)
    # Horizon-blind + auxiliary, training horizons {1,2,4,8} only: tests
    # whether H16 rollouts dilute the sample budget (head remains coherent
    # at any query depth because it never reads the horizon).
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0
    root_distill_weight=0.25; horizon_input=false ;;
  frozen_nonresidual)
    # Frozen recipe with direct (non-residual) MLP prediction: completes the
    # predictor-type x residual 2x2 (the causal predictor preferred direct).
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0
    root_distill_weight=0.25; horizon_input=false; predictor_residual=false ;;
  combo_rank_td)
    # Frozen recipe + expectile-TD value auxiliary: does TD shaping of the
    # latent geometry help the ranking Energy? Planner still scores with the
    # horizon-blind endpoint Energy.
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0
    root_distill_weight=0.25; horizon_input=false; expectile_weight=1
    td_auxiliary=expectile_value ;;
  frozen_k*)
    # Counterfactual-data scaling: frozen recipe with K alternatives per
    # anchor taken from the variant name (frozen_k0, frozen_k8, ...). K=0
    # additionally zeroes counterfactual prediction and the root auxiliary
    # (no alternative roots exist), leaving same-root continuation ranking.
    horizon=8; horizons='[1,2,4,8]'; dense_depth=0; dense_weight=0
    root_distill_weight=0.25; horizon_input=false
    rank_k=${variant#frozen_k}
    if [[ "$rank_k" == "0" ]]; then
      counterfactual_weight=0; root_distill_weight=0; factual_only=true
    fi ;;
  mix_full_16_aux0)
    # Fully pruning-coherent: every integer depth 1..16 is a training
    # horizon, so every beam-pruning Energy query is in-support.
    horizon=16
    horizons='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]'
    dense_depth=0; dense_weight=0; root_distill_weight=0 ;;
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
  fixed_h1_final)
    horizon=1; horizons=null; dense_depth=0; dense_weight=0 ;;
  fixed_h2_final)
    horizon=2; horizons=null; dense_depth=0; dense_weight=0 ;;
  fixed_h8_final)
    horizon=8; horizons=null; dense_depth=0; dense_weight=0 ;;
  fixed_h16_final)
    horizon=16; horizons=null; dense_depth=0; dense_weight=0 ;;
  fixed_h4_rollouts1)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0; rollouts=1 ;;
  fixed_h4_rollouts2)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0; rollouts=2 ;;
  fixed_h4_rollouts8)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0; rollouts=8 ;;
  fixed_h4_rollouts16)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0; rollouts=16 ;;
  fixed_h4_distill0)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0
    root_distill_weight=0 ;;
  fixed_h4_no_endpoint_rank)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0
    horizon_rank_weight=0 ;;
  fixed_h4_no_counterfactual)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0
    counterfactual_weight=0 ;;
  fixed_h4_no_factual_latent)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0
    latent_weight=0 ;;
  fixed_h4_hinge)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0
    ranking_kind=hinge ;;
  fixed_h4_no_chunk_pred)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0
    chunk_weight=0 ;;
  fixed_h4_no_vicreg)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0
    vicreg_weight=0 ;;
  fixed_h4_endpoint_only)
    horizon=4; horizons=null; dense_depth=0; dense_weight=0
    root_distill_weight=0; counterfactual_weight=0; latent_weight=0
    chunk_weight=0 ;;
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
  baseline_td_q)
    # TD-JEPA-adapted SARSA Q competitor: fixed_h4 backbone objectives, no
    # endpoint ranking or root distillation, no geometry rollouts needed.
    horizon=1; horizons=null; dense_depth=0; dense_weight=0; rollouts=1
    rollout_for_h1=false; horizon_rank_weight=0; root_distill_weight=0
    score_mode=td_q; td_q_weight=1 ;;
  baseline_expectile_value)
    # Destrade-style expectile goal-value competitor (arXiv:2601.00844).
    horizon=1; horizons=null; dense_depth=0; dense_weight=0; rollouts=1
    rollout_for_h1=false; horizon_rank_weight=0; root_distill_weight=0
    score_mode=expectile_value; expectile_weight=1 ;;
  baseline_td_jepa)
    # Faithful TD-JEPA competitor (Bagatella et al., arXiv:2510.00739):
    # successor features T(z, u(a), tau(z_0)) with bootstrapped feature TD;
    # z_r is ridge-regressed at plan time (scripts/plan.py).  Same cheap
    # data settings and unchanged backbone objectives as baseline_td_q.
    horizon=1; horizons=null; dense_depth=0; dense_weight=0; rollouts=1
    rollout_for_h1=false; horizon_rank_weight=0; root_distill_weight=0
    score_mode=td_jepa; td_jepa_weight=1 ;;
  baseline_goal_head)
    # Takai et al. (JSAI 2026) GoalHead competitor: g(z_0) trained toward the
    # EMA solved-trajectory endpoint (L2 + cosine); planner scores imagined
    # endpoints by LN-L1 distance to g(z_0).  Non-oracle counterpart of the
    # oracle_goal diagnostic; cheap data settings as baseline_td_q.
    horizon=1; horizons=null; dense_depth=0; dense_weight=0; rollouts=1
    rollout_for_h1=false; horizon_rank_weight=0; root_distill_weight=0
    score_mode=goal_head; goal_head_weight=1 ;;
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
  data.geo_rank_rollout_for_h1="$rollout_for_h1" \
  data.geo_rank_candidate_interface="$candidate_interface" \
  data.geo_rank_k="$rank_k" \
  data.geo_rank_factual_only="$factual_only" \
  data.geo_rank_feasible_k="$feasible_k" \
  data.geo_rank_invalid_k="$invalid_k" \
  data.invalid_action_mode="$invalid_mode" \
  model.geo_rank_score_mode="$score_mode" model.geo_energy_target=distance \
  model.geo_horizon_input="$horizon_input" \
  model.predictor_residual="$predictor_residual" \
  model.geo_td_auxiliary="$td_auxiliary" \
  model.geo_horizon_supervise_prefixes="$prefix_energy" \
  model.dense_rollout_depth="$dense_depth" \
  objective.geo_rank.weight=0 objective.geo_energy_mse.weight=0 \
  objective.geo_advantage_mse.weight="$root_distill_weight" \
  objective.geo_horizon_rank.weight="$horizon_rank_weight" \
  objective.geo_horizon_rank.kind="$ranking_kind" \
  objective.td_q.weight="$td_q_weight" \
  objective.expectile_value.weight="$expectile_weight" \
  objective.td_jepa.weight="$td_jepa_weight" \
  objective.goal_head.weight="$goal_head_weight" \
  objective.counterfactual_state.weight="$counterfactual_weight" \
  objective.latent_pred.weight="$latent_weight" \
  objective.chunk_pred.weight="$chunk_weight" \
  objective.vicreg.weight="$vicreg_weight" \
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
  "$feasible_k" "$invalid_k" "$invalid_mode" "$prefix_energy" \
  "$horizon" "$rollouts" "$horizon_rank_weight" \
  "$counterfactual_weight" "$latent_weight" "$ranking_kind" \
  "$horizon_input" <<'PY'
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
    "fixed_horizon": int(sys.argv[17]),
    "rollouts_per_root": int(sys.argv[18]),
    "endpoint_rank_weight": float(sys.argv[19]),
    "counterfactual_state_weight": float(sys.argv[20]),
    "factual_latent_weight": float(sys.argv[21]),
    "ranking_kind": sys.argv[22],
    "horizon_input": sys.argv[23].lower() == "true",
}, indent=2) + "\n")
PY
