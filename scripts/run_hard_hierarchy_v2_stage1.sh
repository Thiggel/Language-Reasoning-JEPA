#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 WORKER_ID GPU_ID"
  exit 2
fi

worker=$1
gpu=$2
log_dir=research/hard_text/logs/hierarchy_v2_stage1
mkdir -p "$log_dir"

# One-factor screens first, then genuinely nested levels. The flat control has
# identical allocated capacity but disables all high-level objectives.
cells=(
  "flat_control|model.level_spans=[8]|model.level_dims=[32]|objective.high_prediction=0|objective.high_dense=0|objective.reachability=0|objective.high_value=0|objective.macro_prior=0|objective.support=0"
  "l1_s4_d32|model.level_spans=[4]|model.level_dims=[32]"
  "l1_s8_d32|model.level_spans=[8]|model.level_dims=[32]"
  "l1_s12_d32|model.level_spans=[12]|model.level_dims=[32]"
  "l1_s16_d32|model.level_spans=[16]|model.level_dims=[32]"
  "l1_s8_d8|model.level_spans=[8]|model.level_dims=[8]"
  "l1_s8_d16|model.level_spans=[8]|model.level_dims=[16]"
  "l1_s8_d64|model.level_spans=[8]|model.level_dims=[64]"
  "l1_s8_d32_dense1|model.level_spans=[8]|model.level_dims=[32]|model.low_dense_depth=1|model.high_dense_depth=1"
  "l1_s8_d32_dense8|model.level_spans=[8]|model.level_dims=[32]|model.low_dense_depth=8|model.high_dense_depth=8"
  "l1_s8_d32_noreach|model.level_spans=[8]|model.level_dims=[32]|objective.reachability=0"
  "l2_s8_24_d32_16|model.level_spans=[8,24]|model.level_dims=[32,16]|model.variational_levels=[false,false]"
  "l2_s8_32_d32_16|model.level_spans=[8,32]|model.level_dims=[32,16]|model.variational_levels=[false,false]"
  "l2_s8_40_d32_16|model.level_spans=[8,40]|model.level_dims=[32,16]|model.variational_levels=[false,false]"
  "l2_s10_30_d32_16|model.level_spans=[10,30]|model.level_dims=[32,16]|model.variational_levels=[false,false]"
)

for index in "${!cells[@]}"; do
  (( index % 3 == worker )) || continue
  IFS='|' read -r -a fields <<< "${cells[$index]}"
  name="hard_hier_v2_${fields[0]}_s0"
  done_file="$log_dir/$name.done"
  [[ -e "$done_file" ]] && continue
  overrides=()
  for ((j=1; j<${#fields[@]}; j++)); do overrides+=("${fields[$j]}"); done
  {
    .venv/bin/python scripts/train_token_hierarchy_v2.py \
      data=igsm_hard +experiment=hard_hier_v2_screen \
      run_name="$name" seed=0 device="cuda:$gpu" "${overrides[@]}"
    .venv/bin/python scripts/probe_token_hierarchy_v2.py \
      --ckpt "runs/$name/best.pt" --device "cuda:$gpu" --examples 1000
    for mode in flat hierarchy; do
      .venv/bin/python scripts/plan_token_hierarchy_v2.py \
        --hierarchy-ckpt "runs/$name/best.pt" \
        --proposal-ckpt runs/lm_9m_hard/best.pt --device "cuda:$gpu" \
        --mode "$mode" --episodes 50 --beam 8 --branch 4 --horizon 1
    done
    .venv/bin/python scripts/plan_token_hierarchy_v2.py \
      --hierarchy-ckpt "runs/$name/best.pt" \
      --proposal-ckpt runs/lm_9m_hard/best.pt --device "cuda:$gpu" \
      --mode hierarchy --episodes 50 --beam 8 --branch 4 --horizon 1 \
      --oracle-goal
    touch "$done_file"
  } >"$log_dir/$name.log" 2>&1 || touch "$log_dir/$name.failed"
done
