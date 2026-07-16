#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:-runs/intent_hier_k3_d32_det/best.pt}

scripts/run_hierarchy_isolation_cell.sh \
  hierlow_manual_all "$checkpoint" manual_all \
  subgoal_source=oracle_waypoint method=shooting energy=value \
  low_method=discrete low_action_source=all_problem low_horizon=3 \
  low_max_expand=4096 allow_oracle_low_actions=false n_episodes=100

scripts/run_hierarchy_isolation_cell.sh \
  hierlow_manual_cem5 "$checkpoint" manual_cem5 \
  subgoal_source=oracle_waypoint method=shooting energy=value \
  low_method=cem low_horizon=3 low_cem_samples=1200 low_cem_iters=5 \
  low_cem_elites=20 low_cem_variance_ema=0.8 \
  allow_oracle_low_actions=false n_episodes=100

scripts/run_hierarchy_isolation_cell.sh \
  hierlow_manual_cem20 "$checkpoint" manual_cem20 \
  subgoal_source=oracle_waypoint method=shooting energy=value \
  low_method=cem low_horizon=3 low_cem_samples=1200 low_cem_iters=20 \
  low_cem_elites=20 low_cem_variance_ema=0.8 \
  allow_oracle_low_actions=false n_episodes=100

scripts/run_hierarchy_isolation_cell.sh \
  hierlow_model_oracle_cem "$checkpoint" model_oracle_cem \
  subgoal_source=model method=cem energy=oracle_goal high_horizon=2 \
  n_samples=1200 cem_iters=20 n_elites=10 variance_ema=0.9 \
  low_method=cem low_horizon=3 low_cem_samples=1200 low_cem_iters=5 \
  low_cem_elites=20 low_cem_variance_ema=0.8 \
  allow_oracle_low_actions=false n_episodes=100
