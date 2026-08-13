#!/usr/bin/env bash
# Worker-context proposal-coverage ablation (token_igsm).
#
# Charter decision 2 is "can a token worker realize a true future sentence
# waypoint". Four trained variants answered no at ~5% executability, invariant
# to metric, regularizer and learning rate -- so the binding constraint is not
# upstream of the worker. The one factor never varied is the context the
# worker proposes from. This 2x2 separates the model from its context:
#
#   prefix_source x prompt_style
#     canonical      x pinned     -> control, must reproduce greedy ~0.508
#     canonical      x canonical  -> prompt effect at a matched prefix
#     self_generated x pinned     -> should reproduce executability ~0.05
#     self_generated x canonical  -> does format alone restore the worker?
#
# No JEPA checkpoint is loaded and no planner runs: this is frozen-reference
# proposal coverage only, per the contract's labelled-ablation clause.
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

set -euo pipefail

snapshot=${1:?snapshot root required}
source_root=${2:?feature/source root required}
run_dir=${3:?run directory required}
split=${4:-id_test}
max_roots=${5:-128}
population=${6:-32}

python_bin=/home/atuin/c107fa/c107fa12/.venv/bin/python
deps_root=/home/atuin/c107fa/c107fa12/TextJEPA/_deps
mkdir -p "$run_dir"
exec 9>"$run_dir/.writer.lock"
if ! flock -n 9; then
  echo "another job is already writing $run_dir" >&2
  exit 75
fi
printf 'RUNNING\n' > "$run_dir/state"
printf '%s\n' "${SLURM_JOB_ID:-manual}" > "$run_dir/job_id"

finish() {
  code=$?
  printf '%s\n' "$code" > "$run_dir/exit_code"
  if [[ $code -eq 0 ]]; then
    printf 'COMPLETED\n' > "$run_dir/state"
  else
    printf 'FAILED\n' > "$run_dir/state"
  fi
}
trap finish EXIT

if ! command -v module >/dev/null 2>&1 \
    && [[ -r /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load cuda/12.8.1 >/dev/null 2>&1 || true
fi
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/atuin/c107fa/c107fa12/.cache/textjepa/huggingface
export PYTHONPATH="$snapshot/src:$deps_root/hf-xet-1.5.2-py312:$deps_root/language-planning-transformers-5.13.0-minimal-py312"
export TMPDIR=${SLURM_TMPDIR:-/tmp/textjepa-${SLURM_JOB_ID:-manual}}
mkdir -p "$TMPDIR"

cd "$snapshot"
mkdir -p "$run_dir/cells"

# Control first: if it does not reproduce the known greedy ~0.508 the harness
# is wrong and the remaining three cells are not interpretable.
for cell in "canonical:pinned" "self_generated:pinned" \
            "canonical:canonical" "self_generated:canonical"; do
  prefix_source=${cell%%:*}
  prompt_style=${cell##*:}
  output="$run_dir/cells/${split}-${prefix_source}-${prompt_style}.json"
  if [[ -f "$output" ]]; then
    echo "skipping completed cell $output"
    continue
  fi
  echo "=== cell prefix=$prefix_source prompt=$prompt_style"
  "$python_bin" scripts/evaluate_hierarchical_proposal_coverage.py \
    --features "$source_root/features/${split}.pt" \
    --examples "$source_root/splits/${split}.jsonl" \
    --output "$output" \
    --prefix-source "$prefix_source" \
    --prompt-style "$prompt_style" \
    --max-roots "$max_roots" \
    --population "$population" \
    --seed 0 --device cuda --dtype bfloat16
done

"$python_bin" - "$run_dir" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
table = []
for path in sorted((root / "cells").glob("*.json")):
    payload = json.loads(path.read_text())
    table.append({
        "prefix_source": payload["prefix_source"],
        "prompt_style": payload["prompt_style"],
        **payload["metrics"],
    })
(root / "summary.json").write_text(json.dumps(table, indent=2) + "\n")
for row in table:
    print(
        f"{row['prefix_source']:>15} x {row['prompt_style']:<10}"
        f" greedy={row['greedy']:.3f} parse={row['parse_rate']:.3f}"
        f" oracle@32={row.get('oracle@32', float('nan')):.3f}"
        f" unreachable={row['unreachable_rate']:.3f}"
    )
PY
