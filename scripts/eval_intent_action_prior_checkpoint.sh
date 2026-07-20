#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
checkpoint=${2:?source checkpoint}
episodes=${3:-60}
model_dir="$RUN_DIR/model"

test -f "$checkpoint"
mkdir -p "$model_dir"
sha256sum "$checkpoint" > "$model_dir/source_checkpoint.sha256"

for mode in jepa prior top2 top4; do
  extra=()
  case "$mode" in
    prior) extra+=(--prior-only) ;;
    top2) extra+=(--prior-top-m 2) ;;
    top4) extra+=(--prior-top-m 4) ;;
  esac
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_intent_generalization.py" \
    --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
    --lengths 3 6 9 12 15 18 --slacks 0 1 2 4 \
    --episodes "$episodes" --seed 7321 \
    "${extra[@]}" --out "$model_dir/length_curve_${mode}.json"
done

# These are candidate-privileged diagnostics because future feasible action
# menus are supplied by the reference graph.
for depth in 2 4; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_intent_generalization.py" \
    --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
    --lengths 9 12 15 --slacks 0 2 --episodes "$episodes" --seed 7321 \
    --lookahead "$depth" --prior-top-m 4 --allow-oracle-future-actions \
    --out "$model_dir/length_curve_top4_oracle_depth${depth}.json"
done
