#!/usr/bin/env bash
# One-off home-quota cleanup for the 2026-08 admin warning.
#
# Two independent, reversible-by-design steps:
#   1. DELETE 477 duplicate `last.pt` files (~50 GB).  Every one of them sits
#      next to a `best.pt` in the same directory and belongs to a run that has
#      not been touched since 2026-08-10, so no active campaign is affected.
#      All reports and analyses in this repo load `best.pt`.
#   2. MOVE (never delete) every July-2026 checkpoint (~244 GB) to
#      /vol/tmp2, preserving relative paths.  Metrics, logs and result JSONs
#      stay in place, so every recorded number remains readable; only the
#      weights move.  NOTE /vol/tmp2 has no backup.
#      Bulk by subproject: token_igsm 152 GB, sequence_edit 38 GB,
#      intent_phrase 13 GB, other 41 GB.
#
# Usage:  bash scripts/cleanup_home_disk.sh dups     # step 1
#         bash scripts/cleanup_home_disk.sh archive  # step 2 (slow, ~244 GB)
#         bash scripts/cleanup_home_disk.sh plan     # rebuild the file lists
set -uo pipefail

SRC=/vol/home-vol2/ml/laitenbf/TextJEPA
DST=/vol/tmp2/laitenbf/TextJEPA-archive
WORK=${WORK:-/vol/home-vol2/ml/laitenbf/TextJEPA/.cleanup}
PLAN=$WORK/cleanup_plan.txt

plan() {
  mkdir -p "$WORK"
  : > "$PLAN"
  cd "$SRC"
  # Duplicate last.pt: sibling best.pt exists, run untouched since 2026-08-10.
  find runs -name last.pt ! -newermt 2026-08-10 -print0 |
    while IFS= read -r -d '' f; do
      [ -f "$(dirname "$f")/best.pt" ] && printf 'DUP %s %s\n' \
        "$(stat -c%s "$f")" "$f"
    done >> "$PLAN"
  # July-2026 checkpoints (any *.pt), for the archive step.
  find runs -name '*.pt' -newermt 2026-01-01 ! -newermt 2026-08-01 \
    -printf '%s %p\n' >> "$PLAN"
  awk '/^DUP/{n++;s+=$2} END{printf "duplicates: %d files %.1f GB\n",n,s/1e9}' "$PLAN"
  awk '!/^DUP/&&NF==2{n++;s+=$1} END{printf "july:       %d files %.1f GB\n",n,s/1e9}' "$PLAN"
}

dups() {
  [ -s "$PLAN" ] || { echo "run 'plan' first" >&2; exit 2; }
  cd "$SRC"
  awk '/^DUP/{print $3}' "$PLAN" | while read -r f; do
    # Re-check the invariant at delete time, not just at scan time.
    [ -f "$(dirname "$f")/best.pt" ] || { echo "skip (no best.pt): $f"; continue; }
    rm -f -- "$f"
  done
  find runs -path '*/tb/*' -type f -delete
  echo "duplicate checkpoints and tensorboard events removed"
  df -h /vol/home-vol2 | tail -1
}

archive() {
  [ -s "$PLAN" ] || { echo "run 'plan' first" >&2; exit 2; }
  mkdir -p "$DST"
  cd "$SRC"
  awk '!/^DUP/&&NF==2{print $2}' "$PLAN" | while read -r rel; do
    [ -f "$rel" ] || continue
    mkdir -p "$DST/$(dirname "$rel")"
    mv -- "$rel" "$DST/$rel" && echo "MOVED $rel"
  done | tee "$DST/archive.log" | tail -3
  echo "archive complete -> $DST"
  df -h /vol/home-vol2 | tail -1
}

case "${1:-}" in
  plan) plan ;;
  dups) dups ;;
  archive) archive ;;
  *) echo "usage: $0 {plan|dups|archive}" >&2; exit 2 ;;
esac
