#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
name=${2:?cell name}
proposal_mode=${3:?prior or all}
shift 3
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
model_dir="$RUN_DIR/model"
if [[ "$proposal_mode" == "prior" ]]; then
  prior_weight=1.0
else
  prior_weight=0.0
fi
"$python_bin" "${TEXTJEPA_ROOT}/scripts/train_pooled_sentence_jepa.py" \
  "hydra.run.dir=$model_dir" "run_name=$name" "$@" \
  data.train_size=4000 data.val_size=512 \
  "data.n_vars_range=[10,18]" "data.steps_range=[6,12]" \
  train.epochs=6 train.batch_size=8 train.num_workers=0 \
  train.eval_batches=12 train.warmup_steps=200
ckpt="$model_dir/best.pt"
if [[ "$proposal_mode" == "prior" ]]; then
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_pooled_sentence_planning.py" \
    --ckpt "$ckpt" --device cuda:0 --examples 4 --max-tokens 64 \
    --depth 0 --width 8 --planner mpc --score prior \
    --proposals prior --proposal-topk 20 --prior-score-weight 1
fi
for planner in mpc beam; do
  for depth in 1 2 4 8 16; do
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_pooled_sentence_planning.py" \
      --ckpt "$ckpt" --device cuda:0 --examples 4 --max-tokens 64 \
      --depth "$depth" --width 8 --planner "$planner" --score value \
      --proposals "$proposal_mode" --proposal-topk 20 \
      --prior-score-weight "$prior_weight"
  done
done
# Sparse oracle-distance diagnostics isolate model drift from value/ranking error.
for depth in 1 4 16; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_pooled_sentence_planning.py" \
    --ckpt "$ckpt" --device cuda:0 --examples 4 --max-tokens 64 \
    --depth "$depth" --width 8 --planner mpc --score oracle \
    --proposals "$proposal_mode" --proposal-topk 20 \
    --prior-score-weight "$prior_weight"
done
"$python_bin" - "$model_dir" "$RUN_DIR/metrics.json" <<'PY'
import json, pathlib, sys
root, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
checkpoint = __import__("torch").load(root / "best.pt", map_location="cpu", weights_only=False)
payload = {"best_validation": checkpoint["metrics"], "planning": {}}
for path in sorted(root.glob("pooled_*.json")):
    payload["planning"][path.stem] = json.loads(path.read_text())
destination.write_text(json.dumps(payload, indent=2) + "\n")
PY
