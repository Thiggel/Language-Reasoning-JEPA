#!/usr/bin/env bash
set -euo pipefail

checkpoint=$1
prefix=$2
support=${3:-runs/intent_action_support_allstates/best.pt}

for threshold in 2 2.75 3 3.25 4; do
  tag=${threshold//./p}
  scripts/run_hierarchy_isolation_cell.sh \
    "${prefix}_valid_fallback${tag}" "$checkpoint" \
    "valid_fallback${tag}" \
    subgoal_source=discrete_model method=shooting energy=macro_q \
    high_horizon=1 low_method=discrete low_action_source=oracle_feasible \
    low_horizon=3 low_max_expand=4096 allow_oracle_low_actions=true \
    flat_fallback_threshold="$threshold" n_episodes=100
done

# Selector-only controls: execute the first primitive action represented by
# the selected discrete macro.  Comparing these with the subgoal runs above
# isolates high-level choice from low-level subgoal decoding.
scripts/run_hierarchy_isolation_cell.sh \
  "${prefix}_valid_execute_fallback2" "$checkpoint" \
  valid_execute_fallback2 \
  subgoal_source=discrete_model method=shooting energy=macro_q \
  high_horizon=1 low_method=discrete low_action_source=oracle_feasible \
  low_horizon=3 low_max_expand=4096 allow_oracle_low_actions=true \
  discrete_execute_macro=true flat_fallback_threshold=2 n_episodes=100

scripts/run_hierarchy_isolation_cell.sh \
  "${prefix}_valid_execute_fallback3" "$checkpoint" \
  valid_execute_fallback3 \
  subgoal_source=discrete_model method=shooting energy=macro_q \
  high_horizon=1 low_method=discrete low_action_source=oracle_feasible \
  low_horizon=3 low_max_expand=4096 allow_oracle_low_actions=true \
  discrete_execute_macro=true flat_fallback_threshold=3 n_episodes=100

scripts/run_hierarchy_isolation_cell.sh \
  "${prefix}_all_execute_fallback2" "$checkpoint" \
  all_execute_fallback2 low_support_ckpt="$support" \
  subgoal_source=discrete_all method=shooting energy=macro_q \
  high_horizon=1 low_method=discrete low_action_source=all_problem \
  low_horizon=3 low_max_expand=4096 low_support_weight=0.3 \
  discrete_execute_macro=true flat_fallback_threshold=2 \
  allow_oracle_low_actions=false n_episodes=100

scripts/run_hierarchy_isolation_cell.sh \
  "${prefix}_all_execute_fallback3" "$checkpoint" \
  all_execute_fallback3 low_support_ckpt="$support" \
  subgoal_source=discrete_all method=shooting energy=macro_q \
  high_horizon=1 low_method=discrete low_action_source=all_problem \
  low_horizon=3 low_max_expand=4096 low_support_weight=0.3 \
  discrete_execute_macro=true flat_fallback_threshold=3 \
  allow_oracle_low_actions=false n_episodes=100

scripts/run_hierarchy_isolation_cell.sh \
  "${prefix}_all_fallback3" "$checkpoint" all_fallback3 \
  low_support_ckpt="$support" subgoal_source=discrete_all \
  method=shooting energy=macro_q high_horizon=1 low_method=discrete \
  low_action_source=all_problem low_horizon=3 low_max_expand=4096 \
  low_support_weight=0.3 flat_fallback_threshold=3 \
  allow_oracle_low_actions=false n_episodes=100

scripts/run_hierarchy_isolation_cell.sh \
  "${prefix}_cem_fallback3" "$checkpoint" cem_fallback3 \
  low_support_ckpt="$support" subgoal_source=model method=cem \
  energy=macro_q high_horizon=1 n_samples=1200 cem_iters=20 \
  n_elites=10 variance_ema=0.9 density_weight=0.1 \
  learned_support_weight=1.0 low_method=discrete \
  low_action_source=all_problem low_horizon=3 low_max_expand=4096 \
  low_support_weight=0.3 flat_fallback_threshold=3 \
  allow_oracle_low_actions=false n_episodes=100
