#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
checkpoint=${2:?checkpoint}
support_mode=${3:?support mode}
reachability=${4:?reachability 0/1}
feedback_mode=${5:?feedback mode}
flat_horizon=${6:-0}
tag=${7:-oracle_cem}

reach_args=()
if [[ "$reachability" == 1 ]]; then
  reach_args=(--reachability-refine)
fi

flat_args=()
if [[ "$flat_horizon" != 0 ]]; then
  flat_args=(--flat --flat-horizon "$flat_horizon")
fi

"$python_bin" "${TEXTJEPA_ROOT}/scripts/plan_token_hierarchy_oracle_cem.py" \
  --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
  --support-mode "$support_mode" \
  --feedback-mode "$feedback_mode" --feedback-threshold 0.5 \
  --episodes 8 --max-tokens 96 --high-horizon 2 \
  --macro-candidates 256 --macro-iterations 5 --macro-elites 32 \
  --token-candidates 256 --token-iterations 5 --token-elites 32 \
  --token-execution-chunk 1 --reach-topn 8 --reach-budget-scale 0.25 \
  --bank-examples 256 --bank-size 2048 --output-tag "$tag" \
  "${reach_args[@]}" "${flat_args[@]}" > "$RUN_DIR/planner_stdout.json"

cp "$RUN_DIR/planner_stdout.json" "$RUN_DIR/metrics.json"
