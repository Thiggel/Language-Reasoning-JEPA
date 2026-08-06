#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
ckpt=${2:?checkpoint path}
episodes=${3:-2}
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}

if [[ ! -f "$ckpt" ]]; then
  echo "checkpoint not found: $ckpt" >&2
  exit 2
fi

common=(
  --ckpt "$ckpt" --device cuda:0 --episodes "$episodes" --max-tokens 64
  --high-horizon 2 --flat-horizon 32
  --macro-candidates 128 --macro-iterations 5 --macro-elites 16
  --token-candidates 128 --token-iterations 5 --token-elites 16
  --cem-rollout-batch-size 32
  --bank-examples 128 --bank-size 1024 --conditional-bank-k 128
  --support-mode conditional_bank --reachability-refine
  --reach-topn 8 --reach-budget-scale 0.25
  --token-prior-topk 20 --token-prior-weight 0.3
  --tree-width 32 --tree-simulations 128 --macro-tree-topk 8
  --bank-cache "$run_dir/macro_bank.pt"
  --goal-score combined --goal-score-scope top
)

# Keep the hierarchy, macro search, support restriction, seed, and budgets
# fixed.  Only the primitive proposal/search interface changes.
for mode in prior_greedy prior_shooting prior_topk_cem prior_beam prior_astar prior_puct; do
  extra=()
  if [[ "$mode" == "prior_topk_cem" ]]; then
    extra+=(--token-prior-refinements 2)
  fi
  "$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
    --token-proposal "$mode" "${extra[@]}" \
    --out "$run_dir/${mode}.json"
done

"$python_bin" - "$run_dir" "$ckpt" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
checkpoint = pathlib.Path(sys.argv[2])
results = {
    "checkpoint": str(checkpoint),
    "comparison": "matched primitive proposal/search interface",
    "planners": {},
}
for path in sorted(root.glob("prior_*.json")):
    results["planners"][path.stem] = json.loads(path.read_text())
(root / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
PY
