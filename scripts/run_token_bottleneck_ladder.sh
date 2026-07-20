#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
mode=${2:?decomposition, prior-flat, or prior-hierarchy}
ckpt=${3:?checkpoint path}
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}
episodes=${TOKEN_LADDER_EPISODES:-8}
episode_offset=${TOKEN_LADDER_EPISODE_OFFSET:-0}
high_horizon=${TOKEN_LADDER_HIGH_HORIZON:-2}
goal_source=${TOKEN_LADDER_GOAL_SOURCE:-oracle}
value_conditioning=${TOKEN_LADDER_VALUE_CONDITIONING:-goal}

case "$mode" in
  decomposition)
    "$python_bin" scripts/audit_token_closed_loop_decomposition.py \
      --ckpt "$ckpt" --device cuda:0 --examples 8 --max-steps 32 \
      --topk 20 --prior-weights 0.1 1 10 \
      --out "$run_dir/closed_loop_decomposition.json"
    cp "$run_dir/closed_loop_decomposition.json" "$run_dir/metrics.json"
    ;;
  prior-flat|prior-hierarchy)
    common=(
      --ckpt "$ckpt" --device cuda:0 --episodes "$episodes" \
      --episode-offset "$episode_offset" --max-tokens 64
      --high-horizon "$high_horizon" --flat-horizon 1 --token-execution-chunk 1
      --macro-candidates 256 --macro-iterations 10 --macro-elites 32
      --token-candidates 128 --token-iterations 5 --token-elites 16
      --cem-rollout-batch-size 32
      --bank-examples 128 --bank-size 1024 --conditional-bank-k 128
      --support-mode conditional_bank --reachability-refine
      --reach-topn 8 --reach-budget-scale 0.25
      --token-proposal prior_greedy --token-prior-topk 20
      --goal-score combined --goal-score-scope top --value-weight 1
      --goal-source "$goal_source" --value-conditioning "$value_conditioning"
      --bank-cache "$run_dir/macro_bank.pt"
    )
    if [[ "$mode" == prior-flat ]]; then
      "$python_bin" scripts/plan_token_hierarchy_oracle_cem.py \
        "${common[@]}" --flat --out "$run_dir/prior_flat.json"
      cp "$run_dir/prior_flat.json" "$run_dir/metrics.json"
    else
      "$python_bin" scripts/plan_token_hierarchy_oracle_cem.py \
        "${common[@]}" --out "$run_dir/prior_hierarchy.json"
      cp "$run_dir/prior_hierarchy.json" "$run_dir/metrics.json"
    fi
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
