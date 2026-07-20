#!/usr/bin/env bash
# Read-only, bounded storage inventory for sequence-edit cleanup decisions.
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

report="$RUN_DIR/sequence_edit_storage_inventory.log"
: >"$report"

{
  echo "host=$(hostname)"
  echo "date=$(date --iso-8601=seconds)"
  echo "HOME=$HOME"
  echo "WORK=${WORK:-}"
  echo "PROJECT=${PROJECT:-}"
  echo "HPCVAULT=${HPCVAULT:-}"
  df -h "$HOME" 2>/dev/null || true
  quota -s 2>/dev/null || true
} >>"$report"

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

for root in "${roots[@]}"; do
  {
    echo
    echo "===== ROOT $root ====="
    df -h "$root" 2>/dev/null || true
    echo "--- top level size/inodes (bounded depth) ---"
    timeout 900 du -x -h --max-depth=2 "$root" 2>/dev/null \
      | sort -h | tail -80 || echo "du timed out"
    timeout 900 du -x --inodes --max-depth=2 "$root" 2>/dev/null \
      | sort -n | tail -80 || echo "inode du timed out"
    echo "--- sequence-edit candidate directories ---"
    find "$root" -xdev -maxdepth 7 -type d \
      \( -path '*/sequence_edit*' -o -path '*/sequence-edit*' \
         -o -name 'edit-*' -o -name 'edit_*' \) -print 2>/dev/null \
      | head -1000 || true
    echo "--- sequence-edit run trees by size ---"
    while IFS= read -r edit_root; do
      echo "### $edit_root"
      timeout 600 du -x -h --max-depth=2 "$edit_root" 2>/dev/null \
        | sort -h | tail -120 || true
      echo "### failed job directories under $edit_root"
      while IFS= read -r state_file; do
        [[ "$(tr -d '[:space:]' <"$state_file")" == "FAILED" ]] || continue
        job_dir=${state_file%/state}
        du -x -s -h "$job_dir" 2>/dev/null || true
      done < <(find "$edit_root" -xdev -mindepth 3 -maxdepth 3 \
        -type f -name state -print 2>/dev/null)
    done < <(find "$root" -xdev -maxdepth 6 -type d \
      -path '*/runs/autonomy/sequence_edit' -print 2>/dev/null)
    echo "--- failed/temporary/cache candidates (paths only) ---"
    find "$root" -xdev -maxdepth 8 \
      \( -name '*.failed' -o -name 'FAILED' -o -name 'state' \
         -o -name '__pycache__' -o -name '.pytest_cache' \
         -o -name 'wandb' -o -name 'checkpoints' -o -name 'multirun' \
         -o -name 'tmp' -o -name 'slurm-*' \) -print 2>/dev/null \
      | head -3000 || true
  } >>"$report"
done

[[ ${#roots[@]} -gt 0 ]] || {
  echo "No configured storage root was visible" >&2
  exit 3
}
