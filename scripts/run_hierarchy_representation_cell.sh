#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
name=${2:?cell name}
seed=${3:?seed}
spans=${4:?level spans}
dims=${5:?level dims}
level_weights=${6:?level weights}
model_scale=${7:-small}

case "$model_scale" in
  small)
    model_args=(
      model.d_model=256 model.encoder_layers=4 model.predictor_layers=2
      model.n_heads=8 model.ff_mult=4 model.d_action=64 train.batch_size=24
    )
    ;;
  medium)
    model_args=(
      model.d_model=512 model.encoder_layers=6 model.predictor_layers=3
      model.n_heads=8 model.ff_mult=4 model.d_action=96 train.batch_size=12
    )
    ;;
  *)
    echo "unknown model scale: $model_scale" >&2
    exit 2
    ;;
esac

model_dir="$RUN_DIR/model"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/train_token_hierarchy_v2.py" \
  "hydra.run.dir=$model_dir" "run_name=$name" "seed=$seed" \
  data.train_size=6000 data.val_size=1000 train.epochs=3 \
  train.num_workers=4 train.eval_batches=20 train.warmup_steps=200 \
  model.max_len=768 "model.level_spans=$spans" "model.level_dims=$dims" \
  "model.variational_levels=[false]" \
  "model.phase_augmented_levels=[false]" \
  model.low_dense_depth=2 model.high_dense_depth=2 \
  "objective.high_level_weights=$level_weights" \
  "${model_args[@]}"

ckpt="$model_dir/best.pt"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/probe_token_hierarchy_v2.py" \
  --ckpt "$ckpt" --device "${DEVICE:-cuda:0}" --examples 1000 --max-points 12000
"$python_bin" "${TEXTJEPA_ROOT}/scripts/probe_token_hierarchy_symbolic.py" \
  --ckpt "$ckpt" --device "${DEVICE:-cuda:0}" --examples 512
"$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_token_hierarchy_drift.py" \
  --ckpt "$ckpt" --device "${DEVICE:-cuda:0}" --examples 256 --max-horizon 16
"$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_token_hierarchy_gradients.py" \
  --ckpt "$ckpt" --device "${DEVICE:-cuda:0}" --batch-size 16

"$python_bin" - "$model_dir" "$RUN_DIR/metrics.json" <<'PY'
import json, pathlib, sys
root, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
names = {
    "representation": "representation_probes.json",
    "symbolic": "symbolic_linear_probes.json",
    "drift": "predictor_drift_curves.json",
    "gradients": "gradient_diagnostics.json",
}
result = {}
for key, name in names.items():
    path = root / name
    if path.exists():
        result[key] = json.loads(path.read_text())
destination.write_text(json.dumps(result, indent=2) + "\n")
PY
