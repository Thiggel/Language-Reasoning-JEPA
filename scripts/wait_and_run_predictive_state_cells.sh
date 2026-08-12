#!/usr/bin/env bash
set -uo pipefail

# Run deferred run-directory cells on one GPU, in order, once that GPU passes
# the same admission gate the launcher applies. Grünau has no queue, so a round
# that needs more cells than there are momentarily free devices would otherwise
# depend on someone re-running the launcher later.
#
# Admission is re-checked immediately before each cell starts, so the gate keeps
# its meaning: a cell never begins on a device that is busy at that moment.
#
#   wait_and_run_predictive_state_cells.sh <gpu> <job.sh> [<job.sh> ...]

gpu=${1:?gpu index}
shift
[[ $# -gt 0 ]] || { echo "no cells were given" >&2; exit 2; }

checks=${PREDICTIVE_STATE_WAIT_CHECKS:-480}
interval=${PREDICTIVE_STATE_WAIT_INTERVAL:-30}

for job in "$@"; do
  if [[ ! -s "$job" ]]; then
    echo "missing job script: $job" >&2
    continue
  fi
  admitted=0
  for attempt in $(seq 1 "$checks"); do
    reading=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits | awk -F, -v g="$gpu" \
      '{for (i=1;i<=3;i++) gsub(/^[ \t]+|[ \t]+$/,"",$i)} $1==g {print $2" "$3}')
    used=${reading%% *}
    util=${reading##* }
    if [[ -n "$used" && "$used" -lt 1024 && "$util" -lt 10 ]]; then
      echo "admitted gpu $gpu for $job after $attempt checks (used=$used util=$util)"
      admitted=1
      env CUDA_VISIBLE_DEVICES="$gpu" bash "$job"
      break
    fi
    sleep "$interval"
  done
  if [[ "$admitted" -ne 1 ]]; then
    echo "gpu $gpu never became free for $job" >&2
  fi
done
