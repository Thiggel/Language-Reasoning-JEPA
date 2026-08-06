#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
name=${2:?cell name}
shift 2
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
model_dir="$RUN_DIR/model"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/train_pooled_sentence_jepa.py" \
  "hydra.run.dir=$model_dir" "run_name=$name" "$@" \
  data.train_size=10000 data.val_size=2048 \
  "data.n_vars_range=[10,18]" "data.steps_range=[6,12]" \
  train.epochs=20 train.batch_size=8 train.num_workers=0 \
  train.eval_batches=32 train.warmup_steps=1000 train.log_every=100
ckpt="$model_dir/best.pt"

# Tune only the numerical GAR/prior fusion weight on a small calibration set.
for weight in 0 0.1 0.3 1; do
  for depth in 1 4 8 16; do
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_pooled_sentence_planning.py" \
      --ckpt "$ckpt" --device cuda:0 --examples 16 --max-tokens 64 \
      --eval-seed 104731 --output-tag calibration \
      --depth "$depth" --width 8 --planner mpc --score value \
      --proposals prior --proposal-topk 20 --prior-score-weight "$weight"
  done
done
selected_weight=$(
  "$python_bin" - "$model_dir" <<'PY'
import collections, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
scores = collections.defaultdict(list)
for path in root.glob("pooled_calibration_mpc_prior_value_*.json"):
    payload = json.loads(path.read_text())
    scores[float(payload["prior_score_weight"])].append(payload["token_accuracy"])
means = {weight: sum(values) / len(values) for weight, values in scores.items()}
best = max(sorted(means), key=lambda weight: means[weight])
(root / "selected_prior_weight.json").write_text(
    json.dumps({"selected": best, "calibration_mean_by_weight": means}, indent=2) + "\n"
)
print(best)
PY
)

# Independent selection holdout. This is intentionally not the paper test seed.
"$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_pooled_sentence_planning.py" \
  --ckpt "$ckpt" --device cuda:0 --examples 64 --max-tokens 64 \
  --eval-seed 200003 --output-tag holdout \
  --depth 0 --width 8 --planner mpc --score prior \
  --proposals prior --proposal-topk 20 --prior-score-weight 1
for planner in mpc beam; do
  for depth in 1 2 4 8 16; do
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_pooled_sentence_planning.py" \
      --ckpt "$ckpt" --device cuda:0 --examples 64 --max-tokens 64 \
      --eval-seed 200003 --output-tag holdout \
      --depth "$depth" --width 8 --planner "$planner" --score value \
      --proposals prior --proposal-topk 20 --prior-score-weight "$selected_weight"
  done
done
for depth in 1 4 16; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_pooled_sentence_planning.py" \
    --ckpt "$ckpt" --device cuda:0 --examples 64 --max-tokens 64 \
    --eval-seed 200003 --output-tag holdout \
    --depth "$depth" --width 8 --planner mpc --score oracle \
    --proposals prior --proposal-topk 20 --prior-score-weight "$selected_weight"
done

"$python_bin" - "$model_dir" "$RUN_DIR/metrics.json" <<'PY'
import json, pathlib, sys, torch
root, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
payload = {
    "best_epoch": checkpoint["epoch"],
    "best_validation": checkpoint["metrics"],
    "weight_selection": json.loads((root / "selected_prior_weight.json").read_text()),
    "calibration": {}, "holdout": {},
}
for path in sorted(root.glob("pooled_calibration_*.json")):
    payload["calibration"][path.stem] = json.loads(path.read_text())
for path in sorted(root.glob("pooled_holdout_*.json")):
    payload["holdout"][path.stem] = json.loads(path.read_text())
destination.write_text(json.dumps(payload, indent=2) + "\n")
PY
