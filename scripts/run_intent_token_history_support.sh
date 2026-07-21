#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
checkpoint=${2:?source checkpoint}
learning_rate=${3:?learning rate}
history_mode=${4:?aligned or none}
epochs=${5:-5}
train_size=${6:-40000}
episodes=${7:-120}
model_dir="$RUN_DIR/model"

test -f "$checkpoint"
export TMPDIR="/tmp/tj-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR" "$model_dir"
sha256sum "$checkpoint" > "$model_dir/source_checkpoint.sha256"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/train.py" \
  +experiment=paper_causal_j3_catalogue_token_support \
  "hydra.run.dir=$model_dir" \
  "run_name=catalogue_token_history_${history_mode}_lr${learning_rate}" \
  "model.action_support_history_mode=$history_mode" \
  "train.init_ckpt=$checkpoint" \
  "train.lr=$learning_rate" "train.epochs=$epochs" \
  "data.train_size=$train_size" data.val_size=1000 \
  "device=${DEVICE:-cuda:0}"

for length in 6 9; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_action_support.py" \
    --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
    --length "$length" --examples 1000 \
    --out "$model_dir/support_audit_length${length}.json"
done

for support_weight in 1.0 3.0 10.0 30.0; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_intent_generalization.py" \
    --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
    --lengths 6 9 --slacks 0 2 --episodes "$episodes" --seed 7321 \
    --lookahead 1 --proposal-source learned_catalogue \
    --proposal-top-m 4 --proposal-beam-width 4 \
    --proposal-support-weight "$support_weight" --prior-only \
    --out "$model_dir/token_support${support_weight}_prior.json"
done

