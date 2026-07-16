#!/usr/bin/env bash
# Retrain the selected hierarchy depths after the causal reachability audit.
set -u
GPU="${1:-0}"
cd /vol/home-vol2/ml/laitenbf/TextJEPA || exit 1
WAIT=runs/hard_oracle_cem_fast_logs/FOLLOWUP_COMPLETE
while [[ ! -f "$WAIT" ]]; do sleep 30; done
LOGDIR=runs/hard_oracle_cem_fast_logs

train_cell () {
  local name="$1" spans="$2" dims="$3" variational="$4"
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/train_token_hierarchy_v2.py \
    data=igsm_hard +experiment=hard_hier_v2_screen \
    run_name="$name" seed=0 device=cuda:0 \
    model.level_spans="$spans" model.level_dims="$dims" \
    model.variational_levels="$variational" > "$LOGDIR/${name}.log" 2>&1
}

L1NAME=hard_hier_v2_l1_s8_d32_reachhist_s0
L2NAME=hard_hier_v2_l2_s8_32_d32_16_reachhist_s0
L3NAME=hard_hier_v2_l3_s8_32_96_d32_16_8_reachhist_s0
train_cell "$L1NAME" '[8]' '[32]' '[false]'
train_cell "$L2NAME" '[8,32]' '[32,16]' '[false,false]'
train_cell "$L3NAME" '[8,32,96]' '[32,16,8]' '[false,false,false]'

L2="runs/$L2NAME/best.pt"
for mode in unconstrained support_head global_bank conditional_bank gmm conditional_prior; do
  for reach in 0 1; do
    extra=(); [[ "$reach" == 1 ]] && extra=(--reachability-refine)
    CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/plan_token_hierarchy_oracle_cem.py \
      --ckpt "$L2" --device cuda:0 --support-mode "$mode" \
      --episodes 3 --max-tokens 96 --high-horizon 2 \
      --macro-candidates 256 --macro-iterations 5 --macro-elites 32 \
      --token-candidates 128 --token-iterations 4 --token-elites 16 \
      --reach-topn 8 --reach-budget-scale 0.25 --bank-examples 256 \
      --bank-size 2048 --output-tag corrected_fast_e3 "${extra[@]}" \
      > "$LOGDIR/corrected_l2_${mode}_reach${reach}.log" 2>&1
  done
done

for name in "$L1NAME" "$L2NAME" "$L3NAME"; do
  ckpt="runs/$name/best.pt"
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/probe_token_hierarchy_symbolic.py \
    --ckpt "$ckpt" --device cuda:0 --examples 512 \
    > "$LOGDIR/${name}_probes.log" 2>&1
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/audit_token_hierarchy_drift.py \
    --ckpt "$ckpt" --device cuda:0 --examples 256 --max-horizon 16 \
    > "$LOGDIR/${name}_drift.log" 2>&1
done
touch "$LOGDIR/CORRECTED_COMPLETE"
