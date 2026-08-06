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

for mode in jepa prior; do
  extra=()
  [[ "$mode" == prior ]] && extra+=(--prior-only)

  # Prompt-size axis: exact nine-action traces at progressively larger prompts.
  for range in 6:12 18:27 24:36 36:54; do
    lo=${range%%:*}
    hi=${range##*:}
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_intent_generalization.py" \
      --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
      --lengths 9 --n-vars-range "$lo" "$hi" --slacks 0 2 4 \
      --episodes "$episodes" --seed 7321 "${extra[@]}" \
      --out "$model_dir/prompt_axis_l9_nv${lo}_${hi}_${mode}.json"
  done

  # Reasoning-length axis: prompt size is held at 24--36 variables.
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_intent_generalization.py" \
    --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
    --lengths 9 10 11 12 --n-vars-range 24 36 --slacks 0 2 4 \
    --episodes "$episodes" --seed 7321 "${extra[@]}" \
    --out "$model_dir/length_axis_nv24_36_${mode}.json"
done
