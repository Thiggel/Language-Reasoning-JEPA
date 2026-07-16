#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
name=${2:?cell name}
seed=${3:?seed}
boundary_mode=${4:?boundary mode}
level_weights=${5:?level weights}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
model_dir="$RUN_DIR/model"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/train_semantic_token_hierarchy.py" \
  "hydra.run.dir=$model_dir" "run_name=$name" "seed=$seed" \
  "boundary_mode=$boundary_mode" "objective.high_level_weights=$level_weights" \
  data.train_size=6000 data.val_size=1000 \
  "data.n_vars_range=[10,18]" "data.steps_range=[6,12]" \
  train.epochs=3 train.num_workers=0 train.eval_batches=20 train.warmup_steps=200
ckpt="$model_dir/best.pt"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/probe_semantic_token_hierarchy.py" \
  --ckpt "$ckpt" --device cuda:0 --examples 1000 --max-points 12000
"$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_token_hierarchy_gradients.py" \
  --ckpt "$ckpt" --device cuda:0 --batch-size 16
"$python_bin" - "$model_dir" "$RUN_DIR/metrics.json" <<'PY'
import json, pathlib, sys
root, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
result = {}
for key, name in {
    "representation": "semantic_representation_probes.json",
    "gradients": "gradient_diagnostics.json",
}.items():
    result[key] = json.loads((root / name).read_text())
destination.write_text(json.dumps(result, indent=2) + "\n")
PY
