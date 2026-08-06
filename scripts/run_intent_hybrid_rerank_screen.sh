#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
checkpoint=${2:?history-support checkpoint}
seed=${3:?evaluation seed}
episodes=${4:-60}
model_dir="$RUN_DIR/model"

test -f "$checkpoint"
mkdir -p "$model_dir"
sha256sum "$checkpoint" > "$model_dir/source_checkpoint.sha256"

for rerank_weight in 0.0 0.25 0.5 1.0 2.0 4.0; do
  for depth in 1 2 4; do
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_intent_generalization.py" \
      --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
      --lengths 6 9 --slacks 0 2 --episodes "$episodes" --seed "$seed" \
      --lookahead "$depth" --proposal-source learned_catalogue \
      --proposal-top-m 4 --proposal-beam-width 4 \
      --proposal-support-weight 10.0 \
      --proposal-rerank-weight "$rerank_weight" \
      --out "$model_dir/hybrid_weight${rerank_weight}_depth${depth}.json"
  done
done

for depth in 1 2 4; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_intent_generalization.py" \
    --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
    --lengths 6 9 --slacks 0 2 --episodes "$episodes" --seed "$seed" \
    --lookahead "$depth" --proposal-source learned_catalogue \
    --proposal-top-m 4 --proposal-beam-width 4 \
    --proposal-support-weight 10.0 --prior-only \
    --out "$model_dir/prior_endpoint_depth${depth}.json"
done
