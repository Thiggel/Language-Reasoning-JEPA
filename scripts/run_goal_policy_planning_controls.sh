#!/usr/bin/env bash
set -euo pipefail

checkpoint=$1
prefix=$2

scripts/run_hierarchy_isolation_cell.sh \
  "${prefix}_oraclewaypoint" "$checkpoint" oraclewaypoint \
  subgoal_source=oracle_waypoint method=shooting energy=macro_q \
  high_horizon=1 low_method=goal_policy flat_fallback_threshold=3 \
  n_episodes=100

for source in discrete_model discrete_all; do
  scripts/run_hierarchy_isolation_cell.sh \
    "${prefix}_${source}" "$checkpoint" "$source" \
    subgoal_source="$source" method=shooting energy=macro_q \
    high_horizon=1 low_method=goal_policy flat_fallback_threshold=3 \
    low_max_expand=4096 n_episodes=100
done

scripts/run_hierarchy_isolation_cell.sh \
  "${prefix}_cem" "$checkpoint" cem \
  subgoal_source=model method=cem energy=macro_q high_horizon=1 \
  n_samples=1200 cem_iters=20 n_elites=10 variance_ema=0.9 \
  density_weight=0.1 learned_support_weight=1.0 \
  low_method=goal_policy flat_fallback_threshold=3 n_episodes=100
