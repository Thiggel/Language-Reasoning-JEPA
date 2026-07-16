#!/usr/bin/env bash
# Exploratory third-level extension for the matched representation comparison.
set -euo pipefail
gpu=${1:?physical gpu id}
name=hard_hier_v2_l3_s8_32_96_d32_16_8_s0
log=research/hard_text/logs/hierarchy_v2_stage1/$name.log
done=research/hard_text/logs/hierarchy_v2_stage1/$name.done
mkdir -p "$(dirname "$log")"
[[ -f "$done" ]] && exit 0

.venv/bin/python scripts/train_token_hierarchy_v2.py \
  data=igsm_hard +experiment=hard_hier_v2_screen \
  run_name="$name" seed=0 device="cuda:$gpu" \
  model.level_spans='[8,32,96]' model.level_dims='[32,16,8]' \
  model.variational_levels='[false,false,false]' >>"$log" 2>&1
.venv/bin/python scripts/probe_token_hierarchy_v2.py \
  --ckpt "runs/$name/best.pt" --device "cuda:$gpu" \
  --examples 256 --max-points 5000 >>"$log" 2>&1
touch "$done"

