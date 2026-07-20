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
"$python_bin" "${TEXTJEPA_ROOT}/scripts/train_flat_sentence_jepa.py" \
  "hydra.run.dir=$model_dir" "run_name=$name" "$@" \
  data.train_size=4000 data.val_size=512 \
  "data.n_vars_range=[10,18]" "data.steps_range=[6,12]" \
  train.epochs=6 train.batch_size=12 train.num_workers=0 \
  train.eval_batches=16 train.warmup_steps=200
ckpt="$model_dir/best.pt"
for score in oracle value; do
  for depth in 1 2 4 8; do
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_flat_sentence_planning.py" \
      --ckpt "$ckpt" --device cuda:0 --examples 6 --max-tokens 64 \
      --depth "$depth" --width 8 --score "$score" \
      --proposals "$proposal_mode" --proposal-topk 20
  done
done
if [[ "$proposal_mode" == "prior" ]]; then
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_flat_sentence_planning.py" \
    --ckpt "$ckpt" --device cuda:0 --examples 6 --max-tokens 64 \
    --depth 1 --width 1 --score prior --proposals prior --proposal-topk 20
fi
"$python_bin" - "$model_dir" "$RUN_DIR/metrics.json" <<'PY'
import json, pathlib, sys
root, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
payload = {}
for path in sorted(root.glob("flat_sentence_beam_*.json")):
    payload[path.stem] = json.loads(path.read_text())
destination.write_text(json.dumps(payload, indent=2) + "\n")
PY
