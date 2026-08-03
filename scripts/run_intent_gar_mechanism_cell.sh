#!/usr/bin/env bash
# Train one matched GAR-mechanism cell, then run the full frozen audit.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo 'RUN_DIR and TEXTJEPA_ROOT required' >&2
  exit 2
}
py=${1:?python executable}
lr=${2:?learning rate}
bash "$TEXTJEPA_ROOT/scripts/run_intent_fixed_budget_cell.sh" \
  "$py" mlp_jepa "$lr"
bash "$TEXTJEPA_ROOT/scripts/run_intent_decision_audit_cell.sh" \
  "$py" jepa "$RUN_DIR/model/best.pt"
