#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
name=${2:?cell name}
seed=${3:?seed}
dense_depth=${4:?dense rollout depth}
dense_discount=${5:?horizon discount}
high_dense_depth=${6:-$dense_depth}
low_dense_discount=${7:-$dense_discount}
high_dense_discount=${8:-$dense_discount}
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}
model_dir="$run_dir/model"

"$python_bin" scripts/train_token_hierarchy_v2.py \
  "hydra.run.dir=$model_dir" "run_name=$name" "seed=$seed" \
  data.train_size=2000 data.val_size=256 \
  "data.n_vars_range=[10,18]" "data.steps_range=[6,12]" \
  train.epochs=2 train.batch_size=12 train.num_workers=0 \
  train.eval_batches=8 train.warmup_steps=100 \
  model.max_len=768 model.d_model=256 model.encoder_layers=4 \
  model.predictor_layers=2 model.n_heads=8 model.ff_mult=4 model.d_action=64 \
  "model.level_spans=[4,16,64]" "model.level_dims=[32,16,8]" \
  "model.variational_levels=[false]" "model.phase_augmented_levels=[false]" \
  model.distinct_level_states=true model.level_state_encoder_layers=2 \
  "model.low_dense_depth=$dense_depth" "model.high_dense_depth=$high_dense_depth" \
  "objective.dense_discount=$dense_discount" \
  "objective.low_dense_discount=$low_dense_discount" \
  "objective.high_dense_discount=$high_dense_discount" \
  "objective.high_level_weights=[1,1,1]"

ckpt="$model_dir/best.pt"
"$python_bin" scripts/audit_token_hierarchy_drift.py \
  --ckpt "$ckpt" --device cuda:0 --examples 64 --max-horizon 16
"$python_bin" scripts/audit_token_hierarchy_gradients.py \
  --ckpt "$ckpt" --device cuda:0 --batch-size 8
"$python_bin" scripts/probe_token_hierarchy_v2.py \
  --ckpt "$ckpt" --device cuda:0 --examples 256 --max-points 10000

"$python_bin" - "$model_dir" "$run_dir/metrics.json" <<'PY'
import json, pathlib, sys
model, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
result = {}
for name in ("predictor_drift_curves.json", "gradient_diagnostics.json", "representation_probes.json"):
    path = model / name
    if path.exists():
        result[path.stem] = json.loads(path.read_text())
destination.write_text(json.dumps(result, indent=2) + "\n")
PY
