#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
REPORT="$ROOT/research/reports/intent_phrase/2026-08-04-current-findings"
cp "$REPORT"/report_data.json "$HERE/data/results.json"
