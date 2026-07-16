#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:-runs/intent_hfix_d32_full/best.pt}
support=${2:-runs/intent_action_support_allstates/best.pt}
prefix=${3:-d32_}

for energy in value macro_q oracle_goal; do
  name="hierfix_${prefix}macroexec_valid_${energy}"
  scripts/run_hierarchy_isolation_cell.sh \
    "$name" "$checkpoint" "$name" \
    subgoal_source=discrete_model discrete_execute_macro=true \
    method=shooting energy="$energy" high_horizon=1 \
    low_max_expand=4096 n_episodes=100
done

for energy in value macro_q oracle_goal; do
  name="hierfix_${prefix}macroexec_all_${energy}"
  scripts/run_hierarchy_isolation_cell.sh \
    "$name" "$checkpoint" "$name" \
    low_support_ckpt="$support" subgoal_source=discrete_all \
    discrete_execute_macro=true method=shooting energy="$energy" \
    high_horizon=1 low_max_expand=4096 low_support_weight=0.3 \
    allow_oracle_low_actions=false n_episodes=100
done
