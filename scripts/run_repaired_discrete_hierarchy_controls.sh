#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:-runs/intent_hfix_d8_full/best.pt}
support=${2:-runs/intent_action_support_allstates/best.pt}
prefix=${3:-}

# On-manifold upper bound: the reference graph supplies future-feasible macro
# chunks and lower sequences, but the learned high dynamics/value choose them.
for energy in value macro_q oracle_goal; do
  name="hierfix_${prefix}discrete_valid_${energy}"
  scripts/run_hierarchy_isolation_cell.sh \
    "$name" "$checkpoint" "$name" \
    subgoal_source=discrete_model method=shooting energy="$energy" \
    high_horizon=1 low_method=discrete \
    low_action_source=oracle_feasible low_horizon=3 low_max_expand=4096 \
    allow_oracle_low_actions=true n_episodes=100
done

# Deployable text-span planner: enumerate all unrepeated intent phrases, use
# learned feasibility support, and never query future symbolic preconditions.
for energy in value macro_q oracle_goal; do
  name="hierfix_${prefix}discrete_all_${energy}"
  scripts/run_hierarchy_isolation_cell.sh \
    "$name" "$checkpoint" "$name" \
    low_support_ckpt="$support" subgoal_source=discrete_all \
    method=shooting energy="$energy" high_horizon=1 \
    learned_support_weight=0.0 low_method=discrete \
    low_action_source=all_problem low_horizon=3 low_max_expand=4096 \
    low_support_weight=0.3 allow_oracle_low_actions=false n_episodes=100
done
