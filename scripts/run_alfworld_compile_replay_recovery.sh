#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" || -z "${TEXTJEPA_ROOT:-}" ]]; then
  echo "RUN_DIR and TEXTJEPA_ROOT must be supplied by researchctl" >&2
  exit 2
fi
split=${1:?train, val, or test}
raw_input=${2:?absolute preserved raw JSONL path}
[[ -s "$raw_input" ]] || { echo "raw input is absent or empty" >&2; exit 2; }

job_token=${SLURM_JOB_ID:-$$}
worker_tmp=/tmp/awr-$job_token
mkdir -p "$worker_tmp" && chmod 700 "$worker_tmp"
trap 'rm -rf "$worker_tmp"' EXIT
export TMPDIR=$worker_tmp TMP=$worker_tmp TEMP=$worker_tmp

alfworld_python=${ALFWORLD_PYTHON:-/vol/home-vol2/ml/laitenbf/.venv-alfworld/bin/python}
alfworld_data=${ALFWORLD_DATA:-/vol/home-vol2/ml/laitenbf/TextJEPA/data/intent_phrase/alfworld_engine}
project_site=${TEXTJEPA_PROJECT_SITE:-/vol/home-vol2/ml/laitenbf/TextJEPA/.venv/lib64/python3.11/site-packages}
export ALFWORLD_DATA=$alfworld_data
export PYTHONPATH=$TEXTJEPA_ROOT/src:$project_site

compiled=$RUN_DIR/data/compiled/$split.jsonl
"$alfworld_python" "$TEXTJEPA_ROOT/scripts/recompile_alfworld_intent_data.py" \
  --input "$raw_input" --split "$split" --output "$compiled"
"$alfworld_python" "$TEXTJEPA_ROOT/scripts/validate_alfworld_intent_data.py" \
  --dataset "$split=$compiled" --out "$RUN_DIR/metrics.json"
