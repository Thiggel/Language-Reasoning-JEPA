#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?checkpoint}
horizon=${3:?primitive-token MPC horizon}
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}
episodes=${TOKEN_MPC_EPISODES:-2}

"$python_bin" scripts/plan_token_hierarchy_oracle_cem.py \
  --ckpt "$checkpoint" --device cuda:0 --flat \
  --episodes "$episodes" --max-tokens 64 --flat-horizon "$horizon" \
  --token-execution-chunk 1 \
  --token-candidates 128 --token-iterations 5 --token-elites 16 \
  --cem-rollout-batch-size 32 \
  --bank-examples 64 --bank-size 512 --conditional-bank-k 64 \
  --support-mode conditional_bank \
  --token-proposal prior_topk_cem --token-prior-topk 5 \
  --token-prior-refinements 2 --token-prior-weight 0.3 \
  --goal-score combined --goal-score-scope low --value-weight 1 \
  --goal-source oracle --value-conditioning goal \
  --bank-cache "$run_dir/macro_bank.pt" \
  --out "$run_dir/token_mpc_h${horizon}.json"

cp "$run_dir/token_mpc_h${horizon}.json" "$run_dir/metrics.json"
