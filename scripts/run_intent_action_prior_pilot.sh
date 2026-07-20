#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
seed=${2:?seed}
learning_rate=${3:?learning rate}
epochs=${4:?epochs}
warmup_steps=${5:?warmup steps}
train_size=${6:-30000}
episodes=${7:-60}
prior_weight=${8:-1.0}
detach_prior_inputs=${9:-true}
model_dir="$RUN_DIR/model"

export TMPDIR="/tmp/tj-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/train.py" \
  +experiment=paper_causal_j3_action_prior \
  "hydra.run.dir=$model_dir" \
  "run_name=paper_causal_j3_action_prior_lr${learning_rate}_w${prior_weight}_detach${detach_prior_inputs}_s${seed}" \
  "seed=$seed" "train.lr=$learning_rate" "train.epochs=$epochs" \
  "train.warmup_steps=$warmup_steps" "data.train_size=$train_size" \
  "model.action_prior_detach_inputs=$detach_prior_inputs" \
  "objective.action_prior.weight=$prior_weight" \
  data.val_size=1000 "device=${DEVICE:-cuda:0}"

for mode in jepa prior top2 top4; do
  extra=()
  case "$mode" in
    prior) extra+=(--prior-only) ;;
    top2) extra+=(--prior-top-m 2) ;;
    top4) extra+=(--prior-top-m 4) ;;
  esac
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_intent_generalization.py" \
    --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
    --lengths 3 6 9 12 15 18 --slacks 0 1 2 4 \
    --episodes "$episodes" --seed 7321 \
    "${extra[@]}" --out "$model_dir/length_curve_${mode}.json"
done

# These two are deliberately labelled oracle diagnostics: future feasible
# menus come from the reference graph.  They test whether deeper JEPA
# simulation can improve the same top-M proposal interface.
for depth in 2 4; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_intent_generalization.py" \
    --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
    --lengths 9 12 15 --slacks 0 2 --episodes "$episodes" --seed 7321 \
    --lookahead "$depth" --prior-top-m 4 --allow-oracle-future-actions \
    --out "$model_dir/length_curve_top4_oracle_depth${depth}.json"
done
