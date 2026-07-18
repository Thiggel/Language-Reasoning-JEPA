#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
name=${2:?cell name}
seed=${3:?seed}
regression_weight=${4:?GAR advantage MSE weight}
run_search_matrix=${5:?true to run planner matrix}
pairwise_weight=${6:-1}
gar_weight=${7:-0.3}
gar_horizon=${8:-1}
goal_score=${9:-combined}
gar_high_weight=${10:-$gar_weight}
primitive_proposals=${11:-random}
low_policy=${12:-sampled}
continuations=${13:-4}
gar_k=${14:-4}
gar_high_horizon=${15:-$gar_horizon}
gar_beam_width=${16:-4}
gar_beam_branch=${17:-4}
train_size=${18:-2000}
epochs=${19:-2}
batch_size=${20:-12}
gar_low_k=${21:-$gar_k}
gar_high_k=${22:-$gar_k}
gar_objective=${23:-pairwise}
gar_conditional_k=${24:-32}
train_lr=${25:-3e-4}
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}
model_dir="$run_dir/model"

"$python_bin" scripts/train_token_hierarchy_v2.py \
  "hydra.run.dir=$model_dir" "run_name=$name" "seed=$seed" \
  "data.train_size=$train_size" data.val_size=256 \
  "data.n_vars_range=[10,18]" "data.steps_range=[6,12]" \
  "train.epochs=$epochs" "train.batch_size=$batch_size" train.num_workers=0 \
  train.eval_batches=8 train.warmup_steps=100 "train.lr=$train_lr" \
  model.max_len=768 model.d_model=256 model.encoder_layers=4 \
  model.predictor_layers=2 model.n_heads=8 model.ff_mult=4 model.d_action=64 \
  "model.level_spans=[4,16,64]" "model.level_dims=[32,16,8]" \
  "model.variational_levels=[false]" "model.phase_augmented_levels=[false]" \
  model.distinct_level_states=true model.level_state_encoder_layers=2 \
  model.low_dense_depth=4 model.high_dense_depth=4 \
  model.use_token_prior=true model.token_prior_hidden=256 \
  model.token_prior_detach_state=true \
  objective.dense_discount=0.5 "objective.high_level_weights=[1,1,1]" \
  "objective.geo_rank_low=$gar_weight" "objective.geo_rank_high=$gar_high_weight" \
  "objective.geo_rank_level_weights=[1,1,1]" \
  "objective.geo_rank_horizon=$gar_horizon" \
  "objective.geo_rank_low_horizon=$gar_horizon" \
  "objective.geo_rank_high_horizon=$gar_high_horizon" \
  "objective.geo_rank_k=$gar_k" \
  "objective.geo_rank_low_k=$gar_low_k" \
  "objective.geo_rank_high_k=$gar_high_k" \
  "objective.geo_rank_continuations=$continuations" \
  "objective.geo_rank_primitive_proposals=$primitive_proposals" \
  "objective.geo_rank_low_policy=$low_policy" \
  "objective.geo_rank_beam_width=$gar_beam_width" \
  "objective.geo_rank_beam_branch=$gar_beam_branch" \
  objective.geo_rank_label_gap=0.001 \
  "objective.geo_rank_objective=$gar_objective" \
  "objective.geo_rank_pairwise=$pairwise_weight" \
  objective.geo_rank_macro_proposals=conditional \
  "objective.geo_rank_conditional_k=$gar_conditional_k" \
  "objective.geo_rank_regression=$regression_weight" \
  objective.token_prior=1 objective.token_prior_rollout=1 \
  objective.token_prior_rollout_discount=0.5

ckpt="$model_dir/best.pt"
"$python_bin" scripts/audit_token_hierarchy_drift.py \
  --ckpt "$ckpt" --device cuda:0 --examples 64 --max-horizon 16
"$python_bin" scripts/audit_token_hierarchy_gradients.py \
  --ckpt "$ckpt" --device cuda:0 --batch-size 8
"$python_bin" scripts/audit_token_selection.py \
  --ckpt "$ckpt" --device cuda:0 --examples 64 --positions 128
"$python_bin" scripts/audit_token_planner_interface.py \
  --ckpt "$ckpt" --device cuda:0 --examples 16 --positions 64 \
  --topk 20 --goal-horizons 1 4 16 0
"$python_bin" scripts/probe_token_hierarchy_v2.py \
  --ckpt "$ckpt" --device cuda:0 --examples 256 --max-points 10000

common=(
  --ckpt "$ckpt" --device cuda:0 --episodes 2 --max-tokens 64
  --high-horizon 2 --flat-horizon 32
  --macro-candidates 128 --macro-iterations 5 --macro-elites 16
  --token-candidates 128 --token-iterations 5 --token-elites 16
  --cem-rollout-batch-size 32
  --bank-examples 128 --bank-size 1024 --conditional-bank-k 128
  --support-mode conditional_bank --reachability-refine
  --reach-topn 8 --reach-budget-scale 0.25
  --token-prior-topk 20 --token-prior-weight 0.3
  --tree-width 32 --tree-simulations 128 --macro-tree-topk 8
  --bank-cache "$run_dir/macro_bank.pt"
)

for topk in 5 10 20 40; do
  "$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
    --goal-score "$goal_score" --goal-score-scope top --token-proposal prior_topk_cem \
    --token-prior-topk "$topk" --token-prior-refinements 2 \
    --out "$run_dir/top_value_topk${topk}_cem.json"
done
"$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
  --goal-score "$goal_score" --goal-score-scope all --token-proposal prior_topk_cem \
  --token-prior-refinements 2 --out "$run_dir/all_value_topk_cem.json"

if [[ "$run_search_matrix" == "true" ]]; then
  for iterations in 1 10; do
    "$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
      --goal-score "$goal_score" --goal-score-scope top \
      --token-proposal prior_topk_cem --token-prior-refinements 2 \
      --token-iterations "$iterations" \
      --out "$run_dir/top_value_topk20_cem_iter${iterations}.json"
  done
  for token_planner in prior_beam prior_astar prior_puct; do
    "$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
      --goal-score "$goal_score" --goal-score-scope top \
      --token-proposal "$token_planner" \
      --out "$run_dir/top_value_${token_planner}.json"
  done
  for macro_planner in codebook_beam codebook_puct progressive_puct; do
    "$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
      --goal-score "$goal_score" --goal-score-scope top \
      --token-proposal prior_topk_cem --token-prior-refinements 2 \
      --macro-planner "$macro_planner" \
      --out "$run_dir/top_value_${macro_planner}.json"
  done
fi

"$python_bin" - "$model_dir" "$run_dir" <<'PY'
import json, pathlib, sys
model, run = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
result = {}
for name in ("predictor_drift_curves.json", "gradient_diagnostics.json", "token_selection_audit.json", "planner_interface_audit.json", "representation_probes.json"):
    path = model / name
    if path.exists():
        result[path.stem] = json.loads(path.read_text())
for path in sorted(run.glob("*.json")):
    if path.name not in {"metrics.json", "manifest.json", "environment.json", "resolved_config.json", "run_summary.json"}:
        result[path.stem] = json.loads(path.read_text())
(run / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
PY
