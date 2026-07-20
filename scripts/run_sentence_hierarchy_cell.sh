#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
name=${2:?cell name}
value_weight=${3:?planning value weight}
macro_prior_eval=${4:?planning macro-prior weight}
support_eval=${5:?planning support weight}
reachability_eval=${6:?planning reachability weight}
extended_eval=${7:?whether to run the extended planner matrix}
pool_filter=${8:?codebook pool filter}
shift 8
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
model_dir="$RUN_DIR/model"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/train_sentence_hierarchy.py" \
  "hydra.run.dir=$model_dir" "run_name=$name" "$@" \
  data.train_size=4000 data.val_size=512 \
  "data.n_vars_range=[10,18]" "data.steps_range=[6,12]" \
  train.epochs=6 train.batch_size=12 train.num_workers=0 \
  train.eval_batches=16 train.warmup_steps=200
ckpt="$model_dir/best.pt"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_sentence_hierarchy.py" \
  --ckpt "$ckpt" --device cuda:0 --examples 256 --max-points 4000 --gar-k 8
"$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_sentence_prior.py" \
  --ckpt "$ckpt" --device cuda:0 --examples 8 --max-tokens 64
"$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_sentence_hierarchy_planning.py" \
  --ckpt "$ckpt" --device cuda:0 --examples 2 --codebook-examples 256 \
  --macro-support codebook --high-horizon 2 --cem-candidates 256 \
  --cem-updates 10 --cem-elite 32 --codebook-pool 64 --token-topk 20 \
  --pool-filter "$pool_filter" \
  --max-sentence-tokens 32 --max-sentences 8 --max-tokens 64 \
  --goal-weight 1 --value-weight "$value_weight" \
  --macro-prior-weight "$macro_prior_eval" --support-weight "$support_eval" \
  --reachability-weight "$reachability_eval"
if [[ "$extended_eval" == "1" ]]; then
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_sentence_hierarchy_planning.py" \
    --ckpt "$ckpt" --device cuda:0 --examples 2 --codebook-examples 256 \
    --macro-support codebook --high-horizon 2 --cem-candidates 256 \
    --cem-updates 10 --cem-elite 32 --codebook-pool 64 --token-topk 20 \
    --pool-filter "$pool_filter" \
    --max-sentence-tokens 32 --max-sentences 8 --max-tokens 64 \
    --goal-weight 1 --value-weight "$value_weight" \
    --macro-prior-weight "$macro_prior_eval" --support-weight "$support_eval" \
    --reachability-weight "$reachability_eval" --refine-top 4 \
    --refine-weight 1 --output-suffix reach-refined
  for horizon in 1 4; do
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_sentence_hierarchy_planning.py" \
      --ckpt "$ckpt" --device cuda:0 --examples 2 --codebook-examples 256 \
      --macro-support codebook --high-horizon "$horizon" --cem-candidates 256 \
      --cem-updates 10 --cem-elite 32 --codebook-pool 64 --token-topk 20 \
      --pool-filter "$pool_filter" \
      --max-sentence-tokens 32 --max-sentences 8 --max-tokens 64 \
      --goal-weight 1 --value-weight "$value_weight" \
      --macro-prior-weight "$macro_prior_eval" --support-weight "$support_eval" \
      --reachability-weight "$reachability_eval"
  done
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_sentence_hierarchy_planning.py" \
    --ckpt "$ckpt" --device cuda:0 --examples 2 --codebook-examples 256 \
    --macro-support prior --high-horizon 2 --cem-candidates 256 \
    --cem-updates 10 --cem-elite 32 --token-topk 20 \
    --max-sentence-tokens 32 --max-sentences 8 --max-tokens 64 \
    --goal-weight 1 --value-weight "$value_weight" \
    --macro-prior-weight "$macro_prior_eval" --support-weight "$support_eval" \
    --reachability-weight "$reachability_eval"
fi
"$python_bin" - "$model_dir" "$RUN_DIR/metrics.json" <<'PY'
import json, pathlib, sys
root, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
files = {
    "audit": "sentence_hierarchy_audit.json",
    "prior": "sentence_prior_closed_loop.json",
    "planning": "sentence_planning_codebook_h2.json",
}
for path in root.glob("sentence_planning_*.json"):
    files[path.stem] = path.name
destination.write_text(json.dumps({
    key: json.loads((root / name).read_text()) for key, name in files.items()
}, indent=2) + "\n")
PY
