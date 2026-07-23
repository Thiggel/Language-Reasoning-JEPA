#!/usr/bin/env bash
# Read-only inventory restricted to intent-phrase paths and their parent roots.
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
report="$RUN_DIR/intent_phrase_storage_inventory.log"
: >"$report"
printf 'host=%s\ndate=%s\nHOME=%s\nWORK=%s\nPROJECT=%s\nHPCVAULT=%s\n' \
  "$(hostname)" "$(date --iso-8601=seconds)" "$HOME" "${WORK:-}" \
  "${PROJECT:-}" "${HPCVAULT:-}" >>"$report"

roots=()
for root in \
  /vol/home-vol2/ml/laitenbf/TextJEPA \
  "${WORK:-}/TextJEPA" \
  "${PROJECT:-}/TextJEPA" \
  "${HPCVAULT:-}"; do
  [[ -n "$root" && -d "$root" ]] || continue
  real=$(realpath -e "$root")
  [[ " ${roots[*]:-} " == *" $real "* ]] || roots+=("$real")
done

for root in "${roots[@]}"; do
  {
    echo "===== ROOT $root ====="
    df -h "$root" 2>/dev/null || true
    quota -s 2>/dev/null || true
    echo "--- exact intent-phrase trees ---"
    for exact in \
      "$root/runs/autonomy/intent_phrase" \
      "$root/data/intent_phrase" \
      "$root/research/reports/intent_phrase" \
      "$root/research/cycles/intent_phrase"; do
      [[ -d "$exact" ]] && echo "$exact"
    done
    if [[ "$root" == "${HPCVAULT:-/__unset__}" ]]; then
      find "$root" -xdev -maxdepth 6 -type d \
        \( -name 'intent_phrase' -o -name 'intent-phrase' \) \
        -print 2>/dev/null | head -2000
    fi
    echo "--- sizes of intent-phrase run/data trees ---"
    while IFS= read -r path; do
      timeout 900 du -x -h --max-depth=3 "$path" 2>/dev/null \
        | sort -h | tail -160 || echo "du timed out: $path"
      timeout 900 du -x --inodes --max-depth=3 "$path" 2>/dev/null \
        | sort -n | tail -80 || echo "inode du timed out: $path"
    done < <(for exact in \
      "$root/runs/autonomy/intent_phrase" "$root/data/intent_phrase"; do
      [[ -d "$exact" ]] && echo "$exact"
    done)
    echo "--- checkpoint, event, cache, and temporary candidates ---"
    if [[ -d "$root/runs/autonomy/intent_phrase" ]]; then
      find "$root/runs/autonomy/intent_phrase" -xdev \
        \( -path '*/model/*.pt' -o -path '*/model/tb/*' \
           -o -name 'tmp-*' \) -print 2>/dev/null | head -10000
    fi
    if [[ -d "$root/data/intent_phrase" ]]; then
      find "$root/data/intent_phrase" -xdev \
        \( -name '__pycache__' -o -name '.pytest_cache' \) \
        -print 2>/dev/null | head -10000
    fi
  } >>"$report"
done

[[ ${#roots[@]} -gt 0 ]] || {
  echo "No configured storage root was visible" >&2
  exit 3
}
