#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?checkpoint}
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}
common=(
  --ckpt "$checkpoint" --device cuda:0 --episodes 4 --max-tokens 64
  --high-horizon 2 --flat-horizon 32
  --macro-candidates 256 --macro-iterations 5 --macro-elites 32
  --token-candidates 256 --token-iterations 5 --token-elites 32
  --cem-rollout-batch-size 64
  --support-mode conditional_bank --reachability-refine
  --reach-topn 16 --reach-budget-scale 0.25
  --bank-examples 128 --bank-size 1024 --conditional-bank-k 128
  --bank-cache "$run_dir/macro_bank.pt"
)
for scope in low macro; do
  for score in learned_value combined; do
    "$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
      --goal-score "$score" --goal-score-scope "$scope" \
      --out "$run_dir/${scope}_${score}_cem.json"
  done
done

"$python_bin" - "$run_dir" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
result = {}
for path in sorted(root.glob("*_cem.json")):
    result[path.stem] = json.loads(path.read_text())
(root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
PY
