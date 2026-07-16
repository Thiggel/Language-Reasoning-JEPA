#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:-runs/intent_hfix_d32_full/best.pt}
support=${2:-runs/intent_action_support_allstates/best.pt}

for weight in 5.0 10.0 30.0; do
  tag=${weight//./p}
  name="hierfix_d32_macroexec_all_q_fv${tag}"
  scripts/run_hierarchy_isolation_cell.sh \
    "$name" "$checkpoint" "$name" \
    low_support_ckpt="$support" subgoal_source=discrete_all \
    discrete_execute_macro=true discrete_first_value_weight="$weight" \
    method=shooting energy=macro_q high_horizon=1 low_max_expand=4096 \
    low_support_weight=0.3 allow_oracle_low_actions=false n_episodes=100
done
