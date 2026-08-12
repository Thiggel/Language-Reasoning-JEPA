#!/usr/bin/env bash
# Launch the sharded text-only ALFWorld collection for one split.
#
# Collection replays the hand-coded expert plus every counterfactual branch in
# a disposable TextWorld process, so it is CPU bound and takes hours.  Shards
# stripe the seeded game order deterministically, so N shards produce the same
# episode set as one process would, N times faster.
set -euo pipefail

split=${1:?train, val, or test}
size=${2:?total episodes for this split}
shards=${3:-16}
# SHARD_FIRST/SHARD_LAST select which shard indices this host launches, so a
# single striping can be spread over several machines without overlap.
output=${4:?output root (one subdirectory per shard is created)}
counterfactual_k=${COUNTERFACTUAL_K:-2}
invalid_counterfactual_k=${INVALID_COUNTERFACTUAL_K:-2}
teacher_horizon=${TEACHER_HORIZON:-8}
attempts=${COUNTERFACTUAL_ATTEMPTS:-4}
timeout_seconds=${EPISODE_TIMEOUT_SECONDS:-900}
seed=${SEED:-1741}

root=${TEXTJEPA_ROOT:-/vol/home-vol2/ml/laitenbf/TextJEPA}
alfworld_python=${ALFWORLD_PYTHON:-/vol/home-vol2/ml/laitenbf/.venv-alfworld/bin/python}
alfworld_data=${ALFWORLD_DATA:-$root/data/intent_phrase/alfworld_engine}
project_site=${TEXTJEPA_PROJECT_SITE:-$root/.venv/lib64/python3.11/site-packages}
case "$split" in train|val|test) ;; *) echo "invalid split: $split" >&2; exit 2;; esac
if [[ ! -x "$alfworld_python" || ! -d "$alfworld_data/json_2.1.1" ]]; then
  echo "pinned ALFWorld runtime or official data is unavailable" >&2
  exit 2
fi

mkdir -p "$output"
first=${SHARD_FIRST:-0}
last=${SHARD_LAST:-$((shards - 1))}
for ((shard = first; shard <= last; shard++)); do
  shard_dir=$output/shard-$shard
  mkdir -p "$shard_dir"
  worker_tmp=/tmp/aw-$split-$shard-$$
  mkdir -p "$worker_tmp"
  TMPDIR=$worker_tmp TMP=$worker_tmp TEMP=$worker_tmp \
  ALFWORLD_DATA=$alfworld_data \
  PYTHONPATH=$root/src:$project_site \
  setsid nohup "$alfworld_python" \
    "$root/scripts/collect_alfworld_intent_data.py" \
    --data-root "$alfworld_data" --output "$shard_dir/data" \
    --split "$split" "--${split}-size" "$size" \
    --shard-index "$shard" --shard-count "$shards" \
    --counterfactual-k "$counterfactual_k" \
    --invalid-counterfactual-k "$invalid_counterfactual_k" \
    --teacher-horizon "$teacher_horizon" \
    --counterfactual-attempts "$attempts" \
    --episode-timeout-seconds "$timeout_seconds" \
    --seed "$seed" \
    --require-full-counterfactual-coverage \
    > "$shard_dir/log" 2>&1 < /dev/null &
done
echo "launched $split shards $first..$last of $shards under $output"
