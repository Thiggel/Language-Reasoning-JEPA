#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
ckpt=${2:?checkpoint path}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
out="$RUN_DIR/model"
mkdir -p "$out"
for score in oracle value; do
  for depth in 1 2 4 8; do
    for weight in 0.01 0.03 0.1 0.3 1.0; do
      "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_flat_sentence_planning.py" \
        --ckpt "$ckpt" --device cuda:0 --examples 6 --max-tokens 64 \
        --depth "$depth" --width 8 --score "$score" --proposals prior \
        --proposal-topk 20 --prior-score-weight "$weight" --output-dir "$out"
    done
  done
done
for depth in 1 2 4 8; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_flat_sentence_planning.py" \
    --ckpt "$ckpt" --device cuda:0 --examples 6 --max-tokens 64 \
    --depth "$depth" --width 8 --score prior --proposals prior \
    --proposal-topk 20 --output-dir "$out"
done
"$python_bin" - "$out" "$RUN_DIR/metrics.json" <<'PY'
import json, pathlib, sys
root, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
destination.write_text(json.dumps({
    path.stem: json.loads(path.read_text())
    for path in sorted(root.glob("flat_sentence_beam_*.json"))
}, indent=2) + "\n")
PY
