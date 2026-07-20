#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
name=${2:?run name}
weight=${3:?self-rollout weight}
depth=${4:?self-rollout depth}
policy=${5:?greedy or sample}
lr=${6:-3e-4}
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}
model_dir="$run_dir/model"

"$python_bin" scripts/train_token_hierarchy_v2.py \
  "hydra.run.dir=$model_dir" "run_name=$name" seed=0 \
  data.train_size=2000 data.val_size=256 \
  'data.n_vars_range=[10,18]' 'data.steps_range=[6,12]' \
  train.epochs=2 train.batch_size=6 train.num_workers=0 \
  train.eval_batches=8 train.warmup_steps=100 "train.lr=$lr" \
  model.max_len=768 model.d_model=256 model.encoder_layers=4 \
  model.predictor_layers=2 model.n_heads=8 model.ff_mult=4 \
  model.d_action=64 'model.level_spans=[4,16,64]' \
  'model.level_dims=[32,16,8]' 'model.variational_levels=[false]' \
  'model.phase_augmented_levels=[false]' model.distinct_level_states=true \
  model.level_state_encoder_layers=2 model.low_dense_depth=4 \
  model.high_dense_depth=4 model.use_token_prior=true \
  model.token_prior_hidden=256 model.token_prior_detach_state=true \
  objective.dense_discount=0.5 'objective.high_level_weights=[1,1,1]' \
  objective.geo_rank_low=0 objective.geo_rank_high=0 \
  objective.token_prior=1 objective.token_prior_rollout=1 \
  objective.token_prior_rollout_discount=0.5 \
  "objective.token_prior_self_rollout=$weight" \
  "objective.token_prior_self_rollout_depth=$depth" \
  "objective.token_prior_self_rollout_policy=$policy" \
  objective.token_prior_self_rollout_topk=8 \
  objective.token_prior_self_rollout_detach_state=true

ckpt="$model_dir/best.pt"
"$python_bin" scripts/audit_token_selection.py \
  --ckpt "$ckpt" --device cuda:0 --examples 64 --positions 128
"$python_bin" scripts/audit_token_closed_loop_decomposition.py \
  --ckpt "$ckpt" --device cuda:0 --examples 8 --max-steps 32 \
  --topk 20 --prior-weights 1 \
  --out "$run_dir/closed_loop_decomposition.json"
TOKEN_LADDER_EPISODES=8 bash scripts/run_token_bottleneck_ladder.sh \
  "$python_bin" prior-flat "$ckpt"

"$python_bin" - "$model_dir" "$run_dir" <<'PY'
import json, pathlib, sys
model, run = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
result = {
    "token_selection_audit": json.loads((model / "token_selection_audit.json").read_text()),
    "closed_loop_decomposition": json.loads((run / "closed_loop_decomposition.json").read_text()),
    "prior_flat": json.loads((run / "prior_flat.json").read_text()),
}
(run / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
PY
