#!/usr/bin/env bash
# Conservative cleanup: only redundant files inside sequence_edit run trees.
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
execute=${CLEANUP_EXECUTE:-0}
mkdir -p "$RUN_DIR"
manifest="$RUN_DIR/sequence_edit_cleanup_manifest.log"
: >"$manifest"
printf 'host=%s\ndate=%s\nexecute=%s\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" "$execute" >>"$manifest"

roots=()
for root in \
  /vol/home-vol2/ml/laitenbf/TextJEPA \
  "${WORK:-}/TextJEPA" \
  "${PROJECT:-}/TextJEPA" \
  "${HPCVAULT:-}"; do
  [[ -n "$root" && -d "$root" ]] || continue
  duplicate=false
  for seen in "${roots[@]:-}"; do
    [[ "$seen" == "$root" ]] && duplicate=true
  done
  $duplicate || roots+=("$root")
done

freed=0
removed=0
for root in "${roots[@]}"; do
  while IFS= read -r edit_root; do
    canonical=$(realpath -e "$edit_root")
    [[ "$canonical" == */runs/autonomy/sequence_edit ]] || {
      echo "refusing unexpected edit root: $canonical" >&2
      exit 3
    }
    echo "===== $canonical =====" >>"$manifest"
    # last.pt is redundant only when a completed job has a retained best.pt.
    while IFS= read -r last; do
      job=${last%/model/last.pt}
      [[ -f "$job/state" && "$(tr -d '[:space:]' <"$job/state")" == COMPLETED ]] \
        || continue
      [[ -f "$job/model/best.pt" ]] || continue
      real_last=$(realpath -e "$last")
      [[ "$real_last" == "$canonical"/*/model/last.pt ]] || {
        echo "refusing path outside allowlist: $real_last" >&2
        exit 4
      }
      bytes=$(stat -c %s "$real_last")
      printf 'redundant_last_pt\t%s\t%s\n' "$bytes" "$real_last" >>"$manifest"
      if [[ "$execute" == 1 ]]; then
        rm -f -- "$real_last"
      fi
      freed=$((freed + bytes))
      removed=$((removed + 1))
    done < <(find "$canonical" -xdev -type f -path '*/model/last.pt' -print)

    # Job-local temporary directories are disposable after terminal success;
    # scientific logs, metrics, configs, and checkpoints remain untouched.
    while IFS= read -r state_file; do
      [[ "$(tr -d '[:space:]' <"$state_file")" == COMPLETED ]] || continue
      job=${state_file%/state}
      while IFS= read -r temp; do
        real_temp=$(realpath -e "$temp")
        [[ "$real_temp" == "$job"/tmp-* ]] || {
          echo "refusing unexpected temp path: $real_temp" >&2
          exit 5
        }
        bytes=$(du -x -s -B1 "$real_temp" | awk '{print $1}')
        printf 'terminal_tmp\t%s\t%s\n' "$bytes" "$real_temp" >>"$manifest"
        if [[ "$execute" == 1 ]]; then
          rm -rf -- "$real_temp"
        fi
        freed=$((freed + bytes))
        removed=$((removed + 1))
      done < <(find "$job" -xdev -mindepth 1 -maxdepth 1 \
        -type d -name 'tmp-*' -print)
    done < <(find "$canonical" -xdev -mindepth 3 -maxdepth 3 \
      -type f -name state -print)
  done < <(find "$root" -xdev -maxdepth 7 -type d \
    -path '*/runs/autonomy/sequence_edit' -print 2>/dev/null)
done
printf 'candidate_count=%s\ncandidate_bytes=%s\n' \
  "$removed" "$freed" >>"$manifest"
