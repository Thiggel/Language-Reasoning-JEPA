#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
checkpoint=${2:?source checkpoint}
learning_rate=${3:?learning rate}
epochs=${4:-5}
train_size=${5:-40000}
episodes=${6:-60}
model_dir="$RUN_DIR/model"

test -f "$checkpoint"
export TMPDIR="/tmp/tj-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR" "$model_dir"
sha256sum "$checkpoint" > "$model_dir/source_checkpoint.sha256"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/train.py" \
  +experiment=paper_causal_j3_learned_catalogue_heads \
  "hydra.run.dir=$model_dir" \
  "run_name=learned_catalogue_heads_lr${learning_rate}" \
  "train.init_ckpt=$checkpoint" \
  "train.lr=$learning_rate" "train.epochs=$epochs" \
  "data.train_size=$train_size" data.val_size=1000 \
  "device=${DEVICE:-cuda:0}"

for support_weight in 0.5 1.0 2.0; do
  for depth in 1 2 4; do
    for scorer in prior jepa; do
      extra=()
      [[ "$scorer" == prior ]] && extra+=(--prior-only)
      "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_intent_generalization.py" \
        --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
        --lengths 6 9 --slacks 0 2 --episodes "$episodes" --seed 7321 \
        --lookahead "$depth" --proposal-source learned_catalogue \
        --proposal-top-m 4 --proposal-beam-width 4 \
        --proposal-support-weight "$support_weight" \
        "${extra[@]}" \
        --out "$model_dir/catalogue_support${support_weight}_depth${depth}_${scorer}.json"
    done
  done
done
