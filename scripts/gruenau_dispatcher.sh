#!/usr/bin/env bash
# Opportunistic Grünau GPU dispatcher.
#
# Polls every Grünau host for genuinely idle GPUs and starts the next PENDING
# cell of a round on each one.  "Idle" means BOTH low allocated memory and low
# utilisation: a GPU with a large allocation at 0% utilisation is somebody
# else's job between steps, not a free GPU (see CLAUDE.md).
#
# Cells are plain run directories with a `job.sh` and a `state` file.  The
# dispatcher claims a cell by writing CLAIMED before launching, so two
# dispatcher instances cannot start the same cell.  The cell's own job.sh
# writes RUNNING/COMPLETED/FAILED as usual; CUDA_VISIBLE_DEVICES is injected
# at launch time, overriding whatever the file says.
#
# Usage: ROUND=<abs path to round dir> [HOSTS=...] [POLL=300] \
#          bash scripts/gruenau_dispatcher.sh
set -uo pipefail

ROUND=${ROUND:?set ROUND to the round directory}
HOSTS=${HOSTS:-"gruenau1 gruenau2 gruenau7 gruenau8"}
POLL=${POLL:-300}
MEM_FREE_MB=${MEM_FREE_MB:-4000}
UTIL_FREE=${UTIL_FREE:-10}
LOG=$ROUND/dispatcher.log

claim_next() {
  # Prints the path of the next cell it successfully claimed, or nothing.
  for d in "$ROUND"/*/; do
    [ -f "$d/job.sh" ] || continue
    [ "$(cat "$d/state" 2>/dev/null)" = PENDING ] || continue
    # mkdir is atomic: it is the lock.
    if mkdir "$d/.claim" 2>/dev/null; then
      printf 'CLAIMED\n' > "$d/state"
      printf '%s' "$d"
      return 0
    fi
  done
  return 1
}

launch() {
  local host=$1 gpu=$2 cell=$3
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "cd / && CUDA_VISIBLE_DEVICES=$gpu DEVICE=cuda:0 nohup bash '$cell/job.sh' \
       >/dev/null 2>&1 & echo started" >/dev/null 2>&1
}

echo "dispatcher start $(date -u +%FT%TZ) round=$ROUND" >> "$LOG"
while :; do
  pending=$(grep -l PENDING "$ROUND"/*/state 2>/dev/null | wc -l)
  [ "$pending" -eq 0 ] && { echo "no PENDING cells left $(date -u +%FT%TZ)" >> "$LOG"; break; }
  for host in $HOSTS; do
    free_gpus=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
      "nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
         --format=csv,noheader,nounits 2>/dev/null" 2>/dev/null |
      awk -F', ' -v m="$MEM_FREE_MB" -v u="$UTIL_FREE" '$2<m && $3<u {print $1}')
    for gpu in $free_gpus; do
      cell=$(claim_next) || break 2
      [ -n "$cell" ] || break 2
      launch "$host" "$gpu" "$cell"
      echo "$(date -u +%FT%TZ) launched $(basename "$cell") on $host:gpu$gpu" >> "$LOG"
      sleep 60   # let the job allocate before re-reading this host
    done
  done
  sleep "$POLL"
done
