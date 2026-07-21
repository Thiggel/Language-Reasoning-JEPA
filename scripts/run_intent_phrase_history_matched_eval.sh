#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
checkpoint_lr1e3=${2:?phrase-history learning-rate-1e-3 checkpoint}
checkpoint_lr3e3=${3:?phrase-history learning-rate-3e-3 checkpoint}
episodes=${4:-120}
model_dir="$RUN_DIR/model"

test -f "$checkpoint_lr1e3"
test -f "$checkpoint_lr3e3"
mkdir -p "$model_dir"
sha256sum "$checkpoint_lr1e3" "$checkpoint_lr3e3" \
  > "$model_dir/source_checkpoints.sha256"

for tagged_checkpoint in \
  "lr1e3:$checkpoint_lr1e3" "lr3e3:$checkpoint_lr3e3"; do
  tag=${tagged_checkpoint%%:*}
  checkpoint=${tagged_checkpoint#*:}
  for length in 6 9; do
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_action_support.py" \
      --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
      --length "$length" --examples 1000 \
      --out "$model_dir/${tag}_support_audit_length${length}.json"
  done
  for support_weight in 1.0 3.0 10.0 30.0; do
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_intent_generalization.py" \
      --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
      --lengths 6 9 --slacks 0 2 --episodes "$episodes" --seed 7321 \
      --lookahead 1 --proposal-source learned_catalogue \
      --proposal-top-m 4 --proposal-beam-width 4 \
      --proposal-support-weight "$support_weight" --prior-only \
      --out "$model_dir/${tag}_phrase_support${support_weight}_prior.json"
  done
done
