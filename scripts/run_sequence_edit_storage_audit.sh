#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

audit_dir="$RUN_DIR/audit"
cleanup_dir="$RUN_DIR/cleanup-dry-run"
mkdir -p "$audit_dir" "$cleanup_dir"
RUN_DIR="$audit_dir" bash "$TEXTJEPA_ROOT/scripts/audit_sequence_edit_storage.sh"
RUN_DIR="$cleanup_dir" CLEANUP_EXECUTE=0 \
  bash "$TEXTJEPA_ROOT/scripts/cleanup_sequence_edit_storage.sh"
