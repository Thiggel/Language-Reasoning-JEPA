#!/usr/bin/env bash
# Opportunistic Grünau GPU dispatcher.
#
# Grünau GPUs are shared, so "free" is not "empty": what matters is whether
# enough VRAM is left for one of our cells and whether the card is not already
# pinned at full utilisation by someone else.  A card holding another user's
# job at low utilisation with 40 GB spare is perfectly usable; a card with a
# small allocation at 100% utilisation is not.
#
# Placement rule: launch when
#     free_vram >= NEED_MB + MARGIN_MB   and   utilisation <= UTIL_MAX
# The free-VRAM test also stops us stacking two of our own cells on one card:
# once ours is resident its ~NEED_MB shows up as used, so the card no longer
# qualifies.  After each launch we re-query the host rather than trusting the
# pre-launch snapshot, and wait long enough for the new process to allocate.
#
# Cells are run directories with a `job.sh` and a `state` file.  A cell is
# claimed with an atomic mkdir before launch, so parallel dispatchers (or a
# human filling GPUs by hand with the same protocol) cannot double-start one.
#
# Usage: ROUND=<abs path> [HOSTS=...] [NEED_MB=32000] [UTIL_MAX=85] \
#          [POLL=180] bash scripts/gruenau_dispatcher.sh
set -uo pipefail

ROUND=${ROUND:?set ROUND to the round directory}
HOSTS=${HOSTS:-"gruenau1 gruenau2 gruenau7 gruenau8"}
POLL=${POLL:-180}
NEED_MB=${NEED_MB:-32000}      # measured footprint of a 157M-param cell
MARGIN_MB=${MARGIN_MB:-2000}   # never fill a card to the brim
UTIL_MAX=${UTIL_MAX:-85}       # co-schedule politely, don't fight for SMs
SETTLE=${SETTLE:-120}          # seconds for a launched job to allocate
LOG=$ROUND/dispatcher.log

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

usable_gpus() {  # host -> indices with room for us
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" \
    "nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu \
       --format=csv,noheader,nounits 2>/dev/null" 2>/dev/null |
  awk -F', ' -v need="$((NEED_MB + MARGIN_MB))" -v umax="$UTIL_MAX" \
    '($2 - $3) >= need && $4 <= umax {print $1}'
}

claim_next() {
  for d in "$ROUND"/*/; do
    [ -f "$d/job.sh" ] || continue
    [ "$(cat "$d/state" 2>/dev/null)" = PENDING ] || continue
    if mkdir "$d/.claim" 2>/dev/null; then
      printf 'CLAIMED\n' > "$d/state"
      printf '%s' "$d"
      return 0
    fi
  done
  return 1
}

log "dispatcher start round=$ROUND need=${NEED_MB}MB util<=${UTIL_MAX}%"
while :; do
  if ! grep -lq PENDING "$ROUND"/*/state 2>/dev/null; then
    log "no PENDING cells left"; break
  fi
  launched_this_round=0
  for host in $HOSTS; do
    for gpu in $(usable_gpus "$host"); do
      grep -lq PENDING "$ROUND"/*/state 2>/dev/null || break 2
      cell=$(claim_next) || break 2
      [ -n "$cell" ] || break 2
      # -n plus redirecting EVERY remote fd is required: if the backgrounded
      # job keeps the channel's stdout/stderr open, ssh never exits and the
      # dispatcher blocks here forever (observed 2026-08-14: one wedged ssh
      # starved nine PENDING cells while four cards sat free).  `timeout` is
      # the belt for the case a remote fd still leaks.
      # ssh's exit code is NOT a reliable launch signal here: the remote
      # backgrounded job can hold the channel open so ssh is killed by
      # `timeout` (exit 124) even though the job started fine.  Acting on that
      # exit code released a claim on a RUNNING cell and risked a double
      # launch.  So: fire and forget, then ask the cell itself — job.sh writes
      # RUNNING as its first action, which is ground truth.
      timeout 60 ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$host" \
        "cd / && CUDA_VISIBLE_DEVICES=$gpu DEVICE=cuda:0 \
         setsid nohup bash '$cell/job.sh' </dev/null >/dev/null 2>&1 &
         echo ok" </dev/null >/dev/null 2>&1
      started=0
      for _ in $(seq 18); do
        [ "$(cat "$cell/state" 2>/dev/null)" = RUNNING ] && { started=1; break; }
        sleep 5
      done
      if [ "$started" -eq 1 ]; then
        log "launched $(basename "$cell") on $host:gpu$gpu"
        launched_this_round=$((launched_this_round + 1))
        sleep "$SETTLE"
        break   # re-query this host from scratch on the next pass
      else
        log "launch FAILED $(basename "$cell") on $host:gpu$gpu — releasing"
        rmdir "$cell/.claim" 2>/dev/null
        printf 'PENDING\n' > "$cell/state"
      fi
    done
  done
  [ "$launched_this_round" -eq 0 ] && sleep "$POLL"
done
