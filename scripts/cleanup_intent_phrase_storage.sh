#!/usr/bin/env bash
# Conservative cleanup confined to exact intent-phrase run/data trees.
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
execute=${CLEANUP_EXECUTE:-0}
mkdir -p "$RUN_DIR"
manifest="$RUN_DIR/intent_phrase_cleanup_manifest.log"
: >"$manifest"
printf 'host=%s\ndate=%s\nexecute=%s\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" "$execute" >>"$manifest"

roots=()
for root in \
  /vol/home-vol2/ml/laitenbf/TextJEPA \
  "${WORK:-}/TextJEPA" \
  "${PROJECT:-}/TextJEPA"; do
  [[ -n "$root" && -d "$root" ]] || continue
  real=$(realpath -e "$root")
  [[ " ${roots[*]:-} " == *" $real "* ]] || roots+=("$real")
done

freed=0
removed=0
record_and_remove_file() {
  local kind=$1 path=$2 bytes real
  real=$(realpath -e "$path")
  [[ "$real" == */runs/autonomy/intent_phrase/* \
     || "$real" == */data/intent_phrase/* ]] || {
    echo "refusing path outside intent-phrase allowlist: $real" >&2
    exit 4
  }
  bytes=$(stat -c %s "$real")
  printf '%s\t%s\t%s\n' "$kind" "$bytes" "$real" >>"$manifest"
  if [[ "$execute" == 1 ]]; then rm -f -- "$real"; fi
  freed=$((freed + bytes))
  removed=$((removed + 1))
}
is_referenced() {
  local needle=$1 root=$2
  if command -v rg >/dev/null 2>&1; then
    rg -F -q -- "$needle" \
      "$root/projects" "$root/research" "$root/scripts" 2>/dev/null
  else
    grep -R -F -q -- "$needle" \
      "$root/projects" "$root/research" "$root/scripts" 2>/dev/null
  fi
}

for root in "${roots[@]}"; do
  intent_root="$root/runs/autonomy/intent_phrase"
  if [[ -d "$intent_root" ]]; then
    canonical=$(realpath -e "$intent_root")
    [[ "$canonical" == */runs/autonomy/intent_phrase ]] || {
      echo "refusing unexpected run root: $canonical" >&2
      exit 5
    }

    # TensorBoard event streams duplicate retained metrics.csv logs.
    while IFS= read -r event; do
      record_and_remove_file redundant_tensorboard_event "$event"
    done < <(find "$canonical" -xdev -type f \
      -path '*/model/tb/events.out.tfevents.*' -print)

    # A last checkpoint is redundant when the same completed job retains its
    # selected best checkpoint and no source/research document references it.
    while IFS= read -r last; do
      job=${last%/model/last.pt}
      [[ -f "$job/state" \
         && "$(tr -d '[:space:]' <"$job/state")" == COMPLETED ]] || continue
      [[ -f "$job/model/best.pt" ]] || continue
      # Frozen-checkpoint audit E025 used the terminal checkpoints from this
      # invalidated gate; retain them until the corrected gate is analyzed.
      if [[ "$last" == *"/2026-07-22-intent-alfworld-no-prior-matched-overfit-v4/"* ]]; then
        printf 'retained_audit_checkpoint\t0\t%s\n' "$last" >>"$manifest"
        continue
      fi
      if is_referenced "$last" "$root"; then
        printf 'retained_referenced_last\t0\t%s\n' "$last" >>"$manifest"
        continue
      fi
      record_and_remove_file redundant_completed_last "$last"
    done < <(find "$canonical" -xdev -type f \
      -path '*/model/last.pt' -print)

    # Failed or cancelled jobs without any evaluation artifact cannot support
    # a model claim. Preserve configs and logs, remove only checkpoint/event
    # payloads. TIMEOUT and UNKNOWN are retained because they may be resumable.
    while IFS= read -r state_file; do
      state=$(tr -d '[:space:]' <"$state_file")
      [[ "$state" == FAILED || "$state" == CANCELLED ]] || continue
      job=${state_file%/state}
      [[ ! -f "$job/metrics.json" && ! -f "$job/train_metrics.json" ]] \
        || continue
      while IFS= read -r payload; do
        record_and_remove_file invalid_job_model_payload "$payload"
      done < <(find "$job/model" -xdev -type f \
        \( -name '*.pt' -o -name 'events.out.tfevents.*' \) \
        -print 2>/dev/null)
    done < <(find "$canonical" -xdev -mindepth 3 -maxdepth 3 \
      -type f -name state -print)

    # Controller job-local temporary directories are disposable after any
    # terminal state; scientific logs/configs remain.
    while IFS= read -r state_file; do
      state=$(tr -d '[:space:]' <"$state_file")
      [[ "$state" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT)$ ]] || continue
      job=${state_file%/state}
      while IFS= read -r temp; do
        real=$(realpath -e "$temp")
        [[ "$real" == "$job"/tmp-* ]] || {
          echo "refusing unexpected temp path: $real" >&2
          exit 6
        }
        bytes=$(du -x -s -B1 "$real" | awk '{print $1}')
        printf 'terminal_tmp\t%s\t%s\n' "$bytes" "$real" >>"$manifest"
        if [[ "$execute" == 1 ]]; then rm -rf -- "$real"; fi
        freed=$((freed + bytes))
        removed=$((removed + 1))
      done < <(find "$job" -xdev -mindepth 1 -maxdepth 1 \
        -type d -name 'tmp-*' -print)
    done < <(find "$canonical" -xdev -mindepth 3 -maxdepth 3 \
      -type f -name state -print)
  fi

  data_root="$root/data/intent_phrase"
  while [[ -d "$data_root" ]] && IFS= read -r cache; do
    real=$(realpath -e "$cache")
    [[ "$real" == */data/intent_phrase/*/__pycache__ \
       || "$real" == */data/intent_phrase/*/.pytest_cache ]] || {
      echo "refusing unexpected cache path: $real" >&2
      exit 7
    }
    bytes=$(du -x -s -B1 "$real" | awk '{print $1}')
    printf 'intent_data_cache\t%s\t%s\n' "$bytes" "$real" >>"$manifest"
    if [[ "$execute" == 1 ]]; then rm -rf -- "$real"; fi
    freed=$((freed + bytes))
    removed=$((removed + 1))
  done < <(find "$data_root" -xdev -type d \
    \( -name __pycache__ -o -name .pytest_cache \) -print 2>/dev/null)
done

printf 'candidate_count=%s\ncandidate_bytes=%s\n' \
  "$removed" "$freed" >>"$manifest"
