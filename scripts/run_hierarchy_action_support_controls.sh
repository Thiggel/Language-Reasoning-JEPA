#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:-runs/intent_hier_k3_d32_det/best.pt}
support=${2:-runs/intent_action_support_r1/best.pt}

for weight in 0.1 0.3 1.0 3.0; do
  tag=${weight//./p}
  scripts/run_hierarchy_isolation_cell.sh \
    "hierlow_manual_all_sw${tag}" "$checkpoint" "manual_all_sw${tag}" \
    low_support_ckpt="$support" subgoal_source=oracle_waypoint \
    method=shooting energy=value low_method=discrete \
    low_action_source=all_problem low_horizon=3 low_max_expand=4096 \
    low_support_weight="$weight" allow_oracle_low_actions=false \
    n_episodes=100
done

for weight in 0.1 0.3 1.0 3.0; do
  tag=${weight//./p}
  scripts/run_hierarchy_isolation_cell.sh \
    "hierlow_manual_cem_sw${tag}" "$checkpoint" "manual_cem_sw${tag}" \
    low_support_ckpt="$support" subgoal_source=oracle_waypoint \
    method=shooting energy=value low_method=cem low_horizon=3 \
    low_cem_samples=1200 low_cem_iters=20 low_cem_elites=20 \
    low_cem_variance_ema=0.8 low_support_weight="$weight" \
    allow_oracle_low_actions=false n_episodes=100
done
