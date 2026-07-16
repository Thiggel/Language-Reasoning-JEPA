#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_NAME GPU_ID"
  exit 2
fi

name=$1
gpu=$2
ckpt="runs/$name/best.pt"
proposal=runs/lm_9m_hard/best.pt

# Full selected-cell evaluation. These are HWM-scale CEM budgets; no result
# from a smaller engineering smoke is written by this script.
for mode in lm flat hierarchy; do
  .venv/bin/python scripts/plan_token_hierarchy_v2.py \
    --hierarchy-ckpt "$ckpt" --proposal-ckpt "$proposal" \
    --device "cuda:$gpu" --mode "$mode" --episodes 200 \
    --beam 16 --branch 4 --horizon 1
done
for candidates in 900 1000 3000; do
  for iterations in 15 20 40; do
    .venv/bin/python scripts/plan_token_hierarchy_v2.py \
      --hierarchy-ckpt "$ckpt" --proposal-ckpt "$proposal" \
      --device "cuda:$gpu" --mode latent-cem --episodes 100 \
      --beam 16 --branch 4 --horizon 2 \
      --cem-candidates "$candidates" --cem-iterations "$iterations" \
      --cem-elites 100
  done
done

