#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" || -z "${TEXTJEPA_ROOT:-}" ]]; then
  echo "RUN_DIR and TEXTJEPA_ROOT must be supplied by researchctl" >&2
  exit 2
fi

split=${1:?train, val, or test}
size=${2:?number of episodes}
counterfactual_k=${3:-1}
teacher_horizon=${4:-4}
invalid_counterfactual_k=${5:-0}
episode_timeout_seconds=${6:-300}
episode_template=${7:-}
case "$split" in train|val|test) ;; *) echo "invalid split: $split" >&2; exit 2;; esac

job_token=${SLURM_JOB_ID:-$$}
worker_tmp=/tmp/aw-$job_token
mkdir -p "$worker_tmp"
chmod 700 "$worker_tmp"
trap 'rm -rf "$worker_tmp"' EXIT
export TMPDIR=$worker_tmp TMP=$worker_tmp TEMP=$worker_tmp

alfworld_python=${ALFWORLD_PYTHON:-/vol/home-vol2/ml/laitenbf/.venv-alfworld/bin/python}
alfworld_data=${ALFWORLD_DATA:-/vol/home-vol2/ml/laitenbf/TextJEPA/data/intent_phrase/alfworld_engine}
project_site=${TEXTJEPA_PROJECT_SITE:-/vol/home-vol2/ml/laitenbf/TextJEPA/.venv/lib64/python3.11/site-packages}
if [[ ! -x "$alfworld_python" || ! -d "$alfworld_data/json_2.1.1" ]]; then
  echo "pinned ALFWorld runtime or official data is unavailable" >&2
  exit 2
fi
export ALFWORLD_DATA=$alfworld_data
export PYTHONPATH=$TEXTJEPA_ROOT/src:$project_site

size_argument=--${split}-size
output=$RUN_DIR/data
collect_args=(
  --data-root "$alfworld_data" --output "$output" --split "$split"
  "$size_argument" "$size" --counterfactual-k "$counterfactual_k"
  --invalid-counterfactual-k "$invalid_counterfactual_k"
  --teacher-horizon "$teacher_horizon" --counterfactual-attempts 4
  --episode-timeout-seconds "$episode_timeout_seconds"
)
if [[ -n "$episode_template" ]]; then
  collect_args+=(--episode-template "$episode_template")
fi
"$alfworld_python" "$TEXTJEPA_ROOT/scripts/collect_alfworld_intent_data.py" \
  "${collect_args[@]}"

"$alfworld_python" "$TEXTJEPA_ROOT/scripts/validate_alfworld_intent_data.py" \
  --dataset "$split=$output/compiled/$split.jsonl" \
  --out "$RUN_DIR/metrics.json"
