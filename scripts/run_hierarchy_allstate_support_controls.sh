#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:-runs/intent_hier_k3_d32_det/best.pt}
support=${2:-runs/intent_action_support_allstates/best.pt}

for weight in 0.3 1.0; do
  tag=${weight//./p}
  scripts/run_hierarchy_isolation_cell.sh \
    "hierlow_allstate_sw${tag}" "$checkpoint" "allstate_sw${tag}" \
    low_support_ckpt="$support" subgoal_source=oracle_waypoint \
    method=shooting energy=value low_method=discrete \
    low_action_source=all_problem low_horizon=3 low_max_expand=4096 \
    low_support_weight="$weight" allow_oracle_low_actions=false \
    n_episodes=100
done

for threshold in -2.0 0.0; do
  tag=${threshold//./p}; tag=${tag//-/m}
  scripts/run_hierarchy_isolation_cell.sh \
    "hierlow_allstate_st${tag}" "$checkpoint" "allstate_st${tag}" \
    low_support_ckpt="$support" subgoal_source=oracle_waypoint \
    method=shooting energy=value low_method=discrete \
    low_action_source=all_problem low_horizon=3 low_max_expand=4096 \
    low_support_weight=0.3 low_support_threshold="$threshold" \
    allow_oracle_low_actions=false n_episodes=100
done
