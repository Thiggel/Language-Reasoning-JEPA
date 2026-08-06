#!/usr/bin/env bash
set -euo pipefail

python_bin=$1
checkpoint=$2
variant=${3:-standard}
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}

common=(
  --ckpt "$checkpoint" --device cuda:0 --episodes 4 --max-tokens 128
  --high-horizon 2 --flat-horizon 32
  --macro-candidates 1000 --macro-iterations 20 --macro-elites 100
  --token-candidates 1000 --token-iterations 20 --token-elites 100
  --cem-rollout-batch-size 64
  --reach-topn 32 --reach-weight 1.0 --reach-budget-scale 0.25
  --bank-examples 256 --bank-size 2048 --conditional-bank-k 256
)

"$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
  --flat --out "$run_dir/flat_oracle_cem.json"
"$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
  --support-mode conditional_bank --reachability-refine \
  --bank-cache "$run_dir/macro_bank.pt" \
  --out "$run_dir/codebook_reach_oracle_cem.json"
if [[ "$variant" == "with-prior" ]]; then
  "$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
    --support-mode conditional_prior --reachability-refine \
    --bank-cache "$run_dir/macro_bank.pt" \
    --out "$run_dir/prior_reach_oracle_cem.json"
fi
