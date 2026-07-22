#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" || -z "${TEXTJEPA_ROOT:-}" ]]; then
  echo "RUN_DIR and TEXTJEPA_ROOT must be supplied by researchctl" >&2
  exit 2
fi
device=${DEVICE:-cpu}
pilot_root=${ALFWORLD_PILOT_ROOT:-/vol/home-vol2/ml/laitenbf/TextJEPA/data/intent_phrase/alfworld/pilot_v1}
alfworld_data=${ALFWORLD_DATA:-/vol/home-vol2/ml/laitenbf/TextJEPA/data/intent_phrase/alfworld_engine}
eval_python=${ALFWORLD_PYTHON:-/vol/home-vol2/ml/laitenbf/.venv-alfworld/bin/python}
project_site=${TEXTJEPA_PROJECT_SITE:-/vol/home-vol2/ml/laitenbf/TextJEPA/.venv/lib64/python3.11/site-packages}

job_token=${SLURM_JOB_ID:-$$}
worker_tmp=/tmp/awb-$job_token
mkdir -p "$worker_tmp" && chmod 700 "$worker_tmp"
trap 'rm -rf "$worker_tmp"' EXIT
export TMPDIR=$worker_tmp TMP=$worker_tmp TEMP=$worker_tmp
export ALFWORLD_PILOT_ROOT=$pilot_root ALFWORLD_DATA=$alfworld_data
export PYTHONPATH=$TEXTJEPA_ROOT/src:$project_site

for split in train val; do
  "$eval_python" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
    --kind oracle --data-config "$TEXTJEPA_ROOT/configs/data/alfworld_pilot.yaml" \
    --device "$device" --split "$split" --episodes 16 \
    --excess-actions 0 4 --out "$RUN_DIR/oracle_${split}.json"
  "$eval_python" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
    --kind random --data-config "$TEXTJEPA_ROOT/configs/data/alfworld_pilot.yaml" \
    --device "$device" --split "$split" --episodes 16 --seed 7321 \
    --excess-actions 0 4 --out "$RUN_DIR/random_${split}.json"
done
cp "$RUN_DIR/oracle_val.json" "$RUN_DIR/metrics.json"
