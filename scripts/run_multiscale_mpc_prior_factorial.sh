#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
condition=${2:?experiment config}
seed=${3:?seed}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

export TMPDIR="/tmp/tj-${RUN_ID:-multiscale-mpc-$$}"
mkdir -p "$TMPDIR"

case "$condition" in
  edit_multiscale_token) ldad=0; macro=0 ;;
  edit_multiscale_sentence|edit_multiscale_token_sentence) ldad=1; macro=0 ;;
  edit_multiscale_token_sentence_macro) ldad=1; macro=1 ;;
  *) echo "unsupported condition: $condition" >&2; exit 2 ;;
esac

for mode in no_prior detached_prior attached_prior; do
  model_dir="$RUN_DIR/$mode/model"
  mkdir -p "$model_dir"
  base_detach=true
  macro_detach=true
  prior_weight=1
  macro_weight=$macro
  if [[ "$mode" == no_prior ]]; then
    prior_weight=0
    macro_weight=0
  elif [[ "$mode" == attached_prior ]]; then
    base_detach=false
    macro_detach=false
  fi
  "$python_bin" "$TEXTJEPA_ROOT/scripts/train.py" \
    "+experiment=$condition" "hydra.run.dir=$model_dir" \
    "run_name=${RUN_ID}-${mode}" "seed=$seed" "device=${DEVICE:-cuda:0}" \
    model.d_model=320 model.d_action=16 model.d_macro=8 \
    model.base_prior=true model.base_prior_detach_state="$base_detach" \
    model.macro_prior_detach_state="$macro_detach" \
    model.observed_action_ldad="$([[ $ldad == 1 ]] && echo true || echo false)" \
    objective.refinement_prior.weight="$prior_weight" \
    objective.macro_prior_distill.weight="$macro_weight" \
    objective.macro_prior_distill.kind=fixed_variance_mse \
    objective.observed_action_ldad.weight="$ldad" \
    objective.multiscale_vicreg.weight=0 \
    data.train_size=2000 data.val_size=256 data.trajectory_variants=4 \
    data.proposal_pool_k=32 data.proposal_token_pool=prompt_plus_current \
    data.gar_teacher=token_edit_distance \
    train.epochs=1 train.batch_size=8 train.microbatch_size=8 \
    train.num_workers=4 \
    train.lr=0.0003 train.warmup_steps=100 train.eval_batches=8 \
    train.log_every=10

  "$python_bin" "$TEXTJEPA_ROOT/scripts/audit_multiscale_edit.py" \
    --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
    --examples 64 --batches 8 --out "$RUN_DIR/$mode/metrics.json"

  eval_args=()
  if [[ "$mode" == no_prior ]]; then
    eval_args+=(--disable-base-prior --macro-prior-weight 0)
  fi
  "$python_bin" "$TEXTJEPA_ROOT/scripts/eval_multiscale_edit_mpc.py" \
    --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
    --regime id --examples 1 --horizon 1 --beam-width 4 \
    --top-positions 4 --top-tokens 4 --max-candidates 16 \
    --out "$RUN_DIR/$mode/mpc_h1_smoke.json" "${eval_args[@]}"
done
