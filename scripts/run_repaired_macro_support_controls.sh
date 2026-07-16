#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:-runs/intent_hfix_d8_full/best.pt}

for energy in macro_q oracle_goal; do
  for threshold in -2.0 0.0; do
    tag=${threshold//./p}; tag=${tag//-/m}
    name="hierfix_high_${energy}_st${tag}_oraclelow"
    scripts/run_hierarchy_isolation_cell.sh \
      "$name" "$checkpoint" "$name" \
      subgoal_source=model method=cem energy="$energy" high_horizon=1 \
      n_samples=1200 cem_iters=20 n_elites=10 variance_ema=0.9 \
      density_weight=0.1 learned_support_weight=0.0 \
      learned_support_threshold="$threshold" \
      low_method=discrete low_action_source=oracle_feasible low_horizon=3 \
      low_max_expand=4096 allow_oracle_low_actions=true n_episodes=100
  done
done

for spec in "macro_q 0.0 none" "macro_q 1.0 soft" \
            "macro_q 0.0 hard -2.0" "oracle_goal 0.0 hard -2.0"; do
  set -- $spec
  energy=$1; weight=$2; tag=$3; threshold=${4:-}
  args=()
  if [[ -n "$threshold" ]]; then
    args+=(--learned-support-threshold "$threshold")
  fi
  .venv/bin/python scripts/audit_hierarchy_support.py \
    --ckpt "$checkpoint" \
    --out "runs/hieraudit_fix_${energy}_${tag}/support.json" \
    --device cuda:0 --method cem --energy "$energy" --anchors 100 \
    --samples 1200 --iters 20 --elites 10 --high-horizon 1 \
    --density-weight 0.1 --learned-support-weight "$weight" \
    "${args[@]}"
done
