#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
name=${2:?cell name}
epochs=${3:?epochs}
prior_weight=${4:-1}
counterfactual_mse=${5:-0.25}
value_condition=${6:-prompt}
lr=${7:-1e-3}
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}
model_dir="$run_dir/model"

"$python_bin" scripts/train_token_hierarchy_v2.py \
  "hydra.run.dir=$model_dir" "run_name=$name" seed=0 \
  data.train_size=2000 data.val_size=256 \
  "data.n_vars_range=[10,18]" "data.steps_range=[6,12]" \
  "train.epochs=$epochs" train.batch_size=6 train.num_workers=0 \
  train.eval_batches=8 train.warmup_steps=100 "train.lr=$lr" \
  model.max_len=768 model.d_model=256 model.encoder_layers=4 \
  model.predictor_layers=2 model.n_heads=8 model.ff_mult=4 model.d_action=64 \
  "model.level_spans=[4,16,64]" "model.level_dims=[32,16,8]" \
  "model.variational_levels=[false]" "model.phase_augmented_levels=[false]" \
  model.distinct_level_states=true model.level_state_encoder_layers=2 \
  model.low_dense_depth=4 model.high_dense_depth=4 \
  model.use_token_prior=true model.token_prior_hidden=256 \
  model.token_prior_detach_state=true \
  objective.dense_discount=0.5 "objective.high_level_weights=[1,1,1]" \
  objective.token_prior="$prior_weight" objective.token_prior_rollout=1 \
  objective.token_prior_rollout_discount=0.5 \
  objective.geo_rank_low=0.3 objective.geo_rank_high=0.3 \
  "objective.geo_rank_level_weights=[1,1,1]" \
  objective.geo_rank_low_horizon=1 objective.geo_rank_high_horizon=1 \
  objective.geo_rank_low_k=32 objective.geo_rank_high_k=16 \
  objective.geo_rank_continuations=4 \
  objective.geo_rank_primitive_proposals=prior \
  objective.geo_rank_macro_proposals=conditional \
  objective.geo_rank_conditional_k=32 objective.geo_rank_label_gap=0.001 \
  objective.geo_rank_objective=pairwise objective.geo_rank_pairwise=1 \
  objective.geo_rank_regression=0.25 \
  "objective.geo_rank_counterfactual_mse=$counterfactual_mse" \
  "objective.geo_rank_value_condition=$value_condition"

ckpt="$model_dir/best.pt"
"$python_bin" scripts/audit_token_selection.py \
  --ckpt "$ckpt" --device cuda:0 --examples 128 --positions 128
"$python_bin" scripts/audit_token_closed_loop_decomposition.py \
  --ckpt "$ckpt" --device cuda:0 --examples 16 --max-steps 64 \
  --topk 20 --prior-weights 0.1 1 10 \
  --out "$run_dir/closed_loop_decomposition.json"
"$python_bin" - "$model_dir" "$run_dir" <<'PY'
import json, pathlib, sys
model, run = map(pathlib.Path, sys.argv[1:])
result = {
    "token_selection_audit": json.loads((model / "token_selection_audit.json").read_text()),
    "closed_loop_decomposition": json.loads((run / "closed_loop_decomposition.json").read_text()),
}
(run / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
PY
