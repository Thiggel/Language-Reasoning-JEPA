#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:-runs/intent_hfix_d32_full/best.pt}
support=${2:-runs/intent_action_support_allstates/best.pt}

for weight in 0.1 0.3 1.0 3.0; do
  tag=${weight//./p}
  for source in discrete_model discrete_all; do
    suffix=valid; support_args=()
    if [[ "$source" == discrete_all ]]; then
      suffix=all
      support_args=(low_support_ckpt="$support" low_support_weight=0.3)
    fi
    name="hierfix_d32_macroexec_${suffix}_q_fv${tag}"
    scripts/run_hierarchy_isolation_cell.sh \
      "$name" "$checkpoint" "$name" \
      subgoal_source="$source" discrete_execute_macro=true \
      discrete_first_value_weight="$weight" method=shooting energy=macro_q \
      high_horizon=1 low_max_expand=4096 n_episodes=100 \
      "${support_args[@]}"
  done
done
