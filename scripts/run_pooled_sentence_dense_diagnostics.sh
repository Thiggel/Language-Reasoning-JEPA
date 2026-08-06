#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
name=${2:?cell name}
dense_depth=${3:?dense rollout depth}
shift 3
profile=full
if [[ "${1:-}" == "pilot" || "${1:-}" == "full" ]]; then
  profile=$1
  shift
fi
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
if [[ "$profile" == "pilot" ]]; then
  train_size=4000
  val_size=512
  epochs=4
  eval_batches=8
  warmup_steps=200
  calibration_examples=8
  holdout_examples=16
  probe_train=128
  probe_test=64
  decoder_train=256
  decoder_test=64
  decoder_epochs=2
else
  train_size=10000
  val_size=2048
  epochs=20
  eval_batches=32
  warmup_steps=1000
  calibration_examples=16
  holdout_examples=32
  probe_train=512
  probe_test=256
  decoder_train=1024
  decoder_test=256
  decoder_epochs=5
fi
model_dir="$RUN_DIR/model"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/train_pooled_sentence_jepa.py" \
  "hydra.run.dir=$model_dir" "run_name=$name" \
  model.pooling_scope=sentence model.use_prefix_decoder=false \
  model.use_token_prior=true "model.dense_depth=$dense_depth" \
  objective.dense_discount=1.0 train.lr=1e-3 \
  "data.train_size=$train_size" "data.val_size=$val_size" \
  "data.n_vars_range=[10,18]" "data.steps_range=[6,12]" \
  "train.epochs=$epochs" train.batch_size=8 train.eval_batch_size=8 \
  train.gradient_accumulation_steps=1 train.num_workers=0 \
  "train.eval_batches=$eval_batches" "train.warmup_steps=$warmup_steps" \
  train.log_every=100 "$@"
ckpt="$model_dir/best.pt"

# Tune only the GAR/prior fusion coefficient on a disjoint calibration seed.
for weight in 0 0.1 0.3 1; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_pooled_sentence_planning.py" \
    --ckpt "$ckpt" --device cuda:0 --examples "$calibration_examples" --max-tokens 64 \
    --eval-seed 104731 --output-tag calibration \
    --length-mode oracle --depth 4 --width 8 --planner beam --score value \
    --proposals prior --proposal-topk 20 --prior-score-weight "$weight"
done
selected_weight=$(
  "$python_bin" - "$model_dir" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in root.glob("pooled_calibration_beam_prior_value_*.json"):
    payload = json.loads(path.read_text())
    rows.append((payload["token_accuracy"], -payload["prior_score_weight"], payload["prior_score_weight"]))
best = max(rows)[2]
(root / "selected_prior_weight.json").write_text(json.dumps({"selected": best}, indent=2) + "\n")
print(best)
PY
)

# Independent holdout: prior, GAR-value, and privileged oracle-distance scores.
for length_mode in oracle fixed_budget; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_pooled_sentence_planning.py" \
    --ckpt "$ckpt" --device cuda:0 --examples "$holdout_examples" --max-tokens 64 \
    --eval-seed 200003 --output-tag holdout --length-mode "$length_mode" \
    --depth 0 --width 8 --planner beam --score prior \
    --proposals prior --proposal-topk 20 --prior-score-weight 1
  for score in value oracle; do
    for depth in 2 4 8 16; do
      "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_pooled_sentence_planning.py" \
        --ckpt "$ckpt" --device cuda:0 --examples "$holdout_examples" --max-tokens 64 \
        --eval-seed 200003 --output-tag holdout --length-mode "$length_mode" \
        --depth "$depth" --width 8 --planner beam --score "$score" \
        --proposals prior --proposal-topk 20 \
        --prior-score-weight "$selected_weight"
    done
  done
done

# One receding-horizon comparison distinguishes open-loop commitment from MPC.
for score in value oracle; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_pooled_sentence_planning.py" \
    --ckpt "$ckpt" --device cuda:0 --examples "$holdout_examples" --max-tokens 64 \
    --eval-seed 200003 --output-tag holdout --length-mode fixed_budget \
    --depth 4 --width 8 --planner mpc --score "$score" \
    --proposals prior --proposal-topk 20 \
    --prior-score-weight "$selected_weight"
done

"$python_bin" "${TEXTJEPA_ROOT}/scripts/probe_pooled_sentence_states.py" \
  --ckpt "$ckpt" --device cuda:0 --train-examples "$probe_train" --test-examples "$probe_test" \
  --batch-size 16 --output "$model_dir/frozen_sentence_probes.json"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/train_detached_sentence_decoder.py" \
  --ckpt "$ckpt" --device cuda:0 --train-examples "$decoder_train" --test-examples "$decoder_test" \
  --batch-size 64 --epochs "$decoder_epochs" --output "$model_dir/detached_sentence_decoder.json"

"$python_bin" - "$model_dir" "$RUN_DIR/metrics.json" "$profile" <<'PY'
import json, pathlib, sys, torch
root, destination, profile = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
payload = {
    "profile": profile,
    "best_epoch": checkpoint["epoch"],
    "best_validation": checkpoint["metrics"],
    "weight_selection": json.loads((root / "selected_prior_weight.json").read_text()),
    "frozen_sentence_probes": json.loads((root / "frozen_sentence_probes.json").read_text()),
    "detached_sentence_decoder": json.loads((root / "detached_sentence_decoder.json").read_text()),
    "planning": {},
}
for path in sorted(root.glob("pooled_holdout_*.json")):
    payload["planning"][path.stem] = json.loads(path.read_text())
destination.write_text(json.dumps(payload, indent=2) + "\n")
PY
