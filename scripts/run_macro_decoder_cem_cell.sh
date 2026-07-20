#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
detach=${2:?true or false}
decoder_weight=${3:?decoder loss weight}
seed=${4:?seed}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
export TMPDIR="/tmp/tj-${RUN_ID:-macro-decoder-$$}"
mkdir -p "$TMPDIR"

model_dir="$RUN_DIR/model"
"$python_bin" "$TEXTJEPA_ROOT/scripts/train.py" \
  +experiment=edit_multiscale_token_sentence_macro \
  "hydra.run.dir=$model_dir" "run_name=$RUN_ID" "seed=$seed" \
  "device=${DEVICE:-cuda:0}" \
  model.d_model=320 model.d_action=16 model.d_macro=8 model.macro_k=4 \
  model.base_prior=true model.base_prior_detach_state=true \
  model.macro_prior_detach_state=true \
  model.macro_decoder=true model.macro_decoder_detach_inputs="$detach" \
  model.observed_action_ldad=true \
  objective.refinement_prior.weight=1 \
  objective.base_action_value.weight=1 objective.state_goal_distance.weight=1 \
  objective.macro_prior_distill.weight=1 \
  objective.macro_prior_distill.kind=fixed_variance_mse \
  objective.macro_option_reconstruction.weight="$decoder_weight" \
  objective.observed_action_ldad.weight=1 \
  objective.multiscale_vicreg.weight=0 \
  data.train_size=2000 data.val_size=256 data.trajectory_variants=4 \
  data.proposal_pool_k=32 data.proposal_token_pool=prompt_plus_current \
  data.gar_teacher=token_edit_distance \
  train.epochs=1 train.batch_size=8 train.microbatch_size=1 \
  train.lr=0.0003 train.warmup_steps=100 train.eval_batches=8

"$python_bin" "$TEXTJEPA_ROOT/scripts/audit_multiscale_edit.py" \
  --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
  --examples 64 --batches 8 --out "$RUN_DIR/metrics.json"

"$python_bin" "$TEXTJEPA_ROOT/scripts/eval_multiscale_edit_mpc.py" \
  --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
  --regime id --examples 1 --horizon 4 --max-actions 4 \
  --beam-width 4 --top-positions 4 --top-tokens 4 --max-candidates 16 \
  --out "$RUN_DIR/primitive_macro_rerank_smoke.json"

for mode in subgoal decoder_open_loop decoder decoder_refine; do
  "$python_bin" "$TEXTJEPA_ROOT/scripts/eval_hierarchical_edit_cem.py" \
    --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
    --mode "$mode" --regime id --examples 1 --max-actions 4 \
    --high-horizon 1 --low-horizon 4 --cem-candidates 8 \
    --cem-iterations 2 --cem-elites 2 --reachability-topk 2 \
    --beam-width 4 --top-positions 4 --top-tokens 4 \
    --out "$RUN_DIR/${mode}_smoke.json"
done
