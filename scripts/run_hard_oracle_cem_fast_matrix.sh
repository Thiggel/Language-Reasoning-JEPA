#!/usr/bin/env bash
# Fast-signal no-LM hierarchy screen. Usage: script GPU_INDEX
set -u

GPU="${1:-0}"
ROOT=/vol/home-vol2/ml/laitenbf/TextJEPA
cd "$ROOT" || exit 1
LOGDIR=runs/hard_oracle_cem_fast_logs
mkdir -p "$LOGDIR"
FAIL="$LOGDIR/failures.txt"
: > "$FAIL"

run_eval () {
  local name="$1" ckpt="$2" mode="$3" reach="$4"
  shift 4
  local reach_arg=()
  if [[ "$reach" == "1" ]]; then reach_arg=(--reachability-refine); fi
  echo "START $name $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/plan_token_hierarchy_oracle_cem.py \
    --ckpt "$ckpt" --device cuda:0 --support-mode "$mode" \
    --episodes 3 --max-tokens 96 --high-horizon 2 \
    --macro-candidates 256 --macro-iterations 5 --macro-elites 32 \
    --token-candidates 128 --token-iterations 4 --token-elites 16 \
    --reach-topn 8 --reach-budget-scale 0.25 \
    --bank-examples 256 --bank-size 2048 --output-tag fast_e3 \
    "${reach_arg[@]}" "$@" > "$LOGDIR/$name.log" 2>&1
  local status=$?
  if [[ $status -ne 0 ]]; then echo "$name status=$status" >> "$FAIL"; fi
  echo "END $name status=$status $(date --iso-8601=seconds)"
}

L1=runs/hard_hier_v2_l1_s8_d32_s0/best.pt
L2=runs/hard_hier_v2_l2_s8_32_d32_16_s0/best.pt
L3=runs/hard_hier_v2_l3_s8_32_96_d32_16_8_s0/best.pt
FLAT=runs/hard_hier_v2_flat_control_s0/best.pt

for mode in unconstrained support_head global_bank conditional_bank gmm conditional_prior; do
  for reach in 0 1; do
    run_eval "l2_${mode}_reach${reach}" "$L2" "$mode" "$reach"
  done
done

for horizon in 8 32 96; do
  run_eval "flat_h${horizon}" "$FLAT" unconstrained 0 \
    --flat --flat-horizon "$horizon" --output-tag "fast_e3_h${horizon}"
done

for ckpt_name in l1 l3; do
  if [[ "$ckpt_name" == l1 ]]; then ckpt="$L1"; else ckpt="$L3"; fi
  for mode in unconstrained global_bank conditional_prior; do
    for reach in 0 1; do
      run_eval "${ckpt_name}_${mode}_reach${reach}" "$ckpt" "$mode" "$reach"
    done
  done
done

ENSEMBLE=runs/hard_hier_v2_l2_s8_32_d32_16_s0/aligned_macro_ensemble.pt
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/train_aligned_predictor_ensemble.py \
  --ckpt "$L2" --device cuda:0 --members 5 --examples 4096 --epochs 4 \
  --output "$ENSEMBLE" > "$LOGDIR/aligned_ensemble_train.log" 2>&1 || \
  echo "aligned_ensemble_train status=$?" >> "$FAIL"
if [[ -f "$ENSEMBLE" ]]; then
  for reach in 0 1; do
    run_eval "l2_epistemic_reach${reach}" "$L2" unconstrained "$reach" \
      --ensemble-path "$ENSEMBLE" --epistemic-weight 1.0 \
      --output-tag fast_e3_epistemic
  done
fi

for ckpt_name in flat l1 l2 l3; do
  case "$ckpt_name" in
    flat) ckpt="$FLAT" ;;
    l1) ckpt="$L1" ;;
    l2) ckpt="$L2" ;;
    l3) ckpt="$L3" ;;
  esac
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/probe_token_hierarchy_symbolic.py \
    --ckpt "$ckpt" --device cuda:0 --examples 512 \
    > "$LOGDIR/${ckpt_name}_symbolic_probes.log" 2>&1 || \
    echo "${ckpt_name}_symbolic_probes status=$?" >> "$FAIL"
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/audit_token_hierarchy_drift.py \
    --ckpt "$ckpt" --device cuda:0 --examples 256 --max-horizon 16 \
    > "$LOGDIR/${ckpt_name}_drift.log" 2>&1 || \
    echo "${ckpt_name}_drift status=$?" >> "$FAIL"
done

touch "$LOGDIR/COMPLETE"
