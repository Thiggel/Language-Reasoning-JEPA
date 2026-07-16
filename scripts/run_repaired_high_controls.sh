#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:-runs/intent_hfix_d8_full/best.pt}
prefix=${2:-}

run() {
  name=$1 energy=$2 horizon=$3 macro_support=$4
  scripts/run_hierarchy_isolation_cell.sh \
    "$name" "$checkpoint" "$name" \
    subgoal_source=model method=cem energy="$energy" \
    high_horizon="$horizon" n_samples=1200 cem_iters=20 n_elites=10 \
    variance_ema=0.9 density_weight=0.1 \
    learned_support_weight="$macro_support" \
    low_method=discrete low_action_source=oracle_feasible low_horizon=3 \
    low_max_expand=4096 allow_oracle_low_actions=true n_episodes=100
}

run "hierfix_${prefix}high_value_h1_oraclelow" value 1 0.0
run "hierfix_${prefix}high_value_h2_oraclelow" value 2 0.0
run "hierfix_${prefix}high_q_h1_oraclelow" macro_q 1 0.0
run "hierfix_${prefix}high_q_h1_support_oraclelow" macro_q 1 1.0
run "hierfix_${prefix}high_oracle_h1_oraclelow" oracle_goal 1 0.0
run "hierfix_${prefix}high_oracle_h2_oraclelow" oracle_goal 2 0.0
