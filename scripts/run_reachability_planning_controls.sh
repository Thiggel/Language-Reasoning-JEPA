#!/usr/bin/env bash
set -euo pipefail

checkpoint=$1
prefix=$2
support=${3:-runs/intent_action_support_allstates/best.pt}

scripts/run_hierarchy_isolation_cell.sh \
  "${prefix}_discrete_valid_q_subgoal" "$checkpoint" discrete_valid_q_subgoal \
  subgoal_source=discrete_model method=shooting energy=macro_q \
  high_horizon=1 low_method=discrete low_action_source=oracle_feasible \
  low_horizon=3 low_max_expand=4096 allow_oracle_low_actions=true \
  n_episodes=100

scripts/run_hierarchy_isolation_cell.sh \
  "${prefix}_discrete_all_q_subgoal" "$checkpoint" discrete_all_q_subgoal \
  low_support_ckpt="$support" subgoal_source=discrete_all \
  method=shooting energy=macro_q high_horizon=1 low_method=discrete \
  low_action_source=all_problem low_horizon=3 low_max_expand=4096 \
  low_support_weight=0.3 allow_oracle_low_actions=false n_episodes=100

scripts/run_hierarchy_isolation_cell.sh \
  "${prefix}_cem_q_oraclelow" "$checkpoint" cem_q_oraclelow \
  subgoal_source=model method=cem energy=macro_q high_horizon=1 \
  n_samples=1200 cem_iters=20 n_elites=10 variance_ema=0.9 \
  density_weight=0.1 learned_support_weight=1.0 \
  low_method=discrete low_action_source=oracle_feasible low_horizon=3 \
  low_max_expand=4096 allow_oracle_low_actions=true n_episodes=100

scripts/run_hierarchy_isolation_cell.sh \
  "${prefix}_cem_q_deploy" "$checkpoint" cem_q_deploy \
  low_support_ckpt="$support" subgoal_source=model method=cem \
  energy=macro_q high_horizon=1 n_samples=1200 cem_iters=20 \
  n_elites=10 variance_ema=0.9 density_weight=0.1 \
  learned_support_weight=1.0 low_method=discrete \
  low_action_source=all_problem low_horizon=3 low_max_expand=4096 \
  low_support_weight=0.3 allow_oracle_low_actions=false n_episodes=100
