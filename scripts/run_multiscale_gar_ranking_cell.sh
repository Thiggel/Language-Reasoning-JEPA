#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
proposal_pool_k=${2:?counterfactual candidate count}
pairwise_weight=${3:?pairwise ranking weight}
learning_rate=${4:-0.0003}
seed=${5:-0}
train_size=${6:-512}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
export TMPDIR="/tmp/tj-${RUN_ID:-multiscale-gar-ranking-$$}"
mkdir -p "$TMPDIR"

model_dir="$RUN_DIR/model"
"$python_bin" "$TEXTJEPA_ROOT/scripts/train.py" \
  +experiment=edit_multiscale_token \
  "hydra.run.dir=$model_dir" "run_name=$RUN_ID" "seed=$seed" \
  "device=${DEVICE:-cuda:0}" \
  model.d_model=320 model.d_action=16 \
  model.base_prior=true model.base_prior_detach_state=true \
  objective.refinement_prior.weight=1 \
  objective.base_action_value.weight=1 \
  objective.base_action_value.regression_kind=mse \
  objective.base_action_value.regression_weight=0.25 \
  "objective.base_action_value.pairwise_weight=$pairwise_weight" \
  objective.base_action_value.margin=0.5 \
  objective.base_action_value.label_gap=0 \
  objective.state_goal_distance.weight=1 \
  data.train_size="$train_size" data.val_size=256 \
  data.trajectory_variants=4 \
  data.proposal_pool_k="$proposal_pool_k" \
  data.proposal_token_pool=prompt_plus_current \
  data.gar_teacher=token_edit_distance \
  train.epochs=1 train.batch_size=8 train.microbatch_size=8 \
  train.num_workers=4 train.lr="$learning_rate" \
  train.warmup_steps=100 train.eval_batches=8 train.log_every=10

"$python_bin" "$TEXTJEPA_ROOT/scripts/audit_multiscale_edit.py" \
  --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
  --examples 64 --batches 8 --out "$RUN_DIR/metrics.json"

"$python_bin" "$TEXTJEPA_ROOT/scripts/eval_multiscale_edit_mpc.py" \
  --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
  --regime id --examples 1 --horizon 1 --beam-width 4 \
  --top-positions 4 --top-tokens 4 --max-candidates 16 \
  --out "$RUN_DIR/mpc_h1_smoke.json"
