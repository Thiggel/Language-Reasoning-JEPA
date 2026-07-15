#!/usr/bin/env bash
# Immutable evaluation-only worker for completed overnight token-prior cells.
set -euo pipefail

NAME="${1:?cell name}"
SEED="${2:-0}"
ROOT="${TEXTJEPA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${TEXTJEPA_PYTHON:-$ROOT/.venv/bin/python}"
RUN_NAME="overnight_token_prior_${NAME}_s${SEED}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
LOG_DIR="$ROOT/runs/overnight_token_prior_logs"
CKPT="$RUN_DIR/best.pt"
[[ -s "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 2; }
mkdir -p "$LOG_DIR"
rm -f "$RUN_DIR/EVAL_COMPLETE" "$RUN_DIR/EVAL_FAILED"
cd "$ROOT"

COMMON=(
  --ckpt "$CKPT" --device cuda:0 --support-mode conditional_bank
  --feedback-mode adaptive --feedback-threshold 0.5
  --token-execution-chunk 1 --episodes 3 --max-tokens 64
  --high-horizon 2 --macro-candidates 512 --macro-iterations 8
  --macro-elites 64 --token-candidates 256 --token-iterations 5
  --token-elites 32 --bank-examples 256 --bank-size 2048
)

run_one() {
  local mode="$1" weight="$2" temperature="$3" topk="$4" reach="$5"
  local reach_tag=""
  local reach_args=()
  if [[ "$reach" == 1 ]]; then
    reach_tag="_reach"
    reach_args=(--reachability-refine --reach-topn 16 --reach-budget-scale 0.25)
  fi
  local tag="${NAME}_${mode}_w${weight}_t${temperature}_k${topk}${reach_tag}"
  local result="$RUN_DIR/oracle_cem_conditional_bank_reach${reach}_${tag}.json"
  if [[ -s "$result" ]]; then
    echo "SKIP existing $result"
    return
  fi
  "$PYTHON" scripts/plan_token_hierarchy_oracle_cem.py \
    "${COMMON[@]}" --token-proposal "$mode" \
    --token-prior-weight "$weight" --token-prior-temperature "$temperature" \
    --token-prior-topk "$topk" "${reach_args[@]}" --output-tag "$tag" \
    > "$LOG_DIR/${tag}.log" 2>&1
}

if ! {
  run_one uniform 0.0 1.0 0 0
  run_one prior_greedy 0.0 1.0 0 0
  run_one prior_shooting 0.0 0.8 32 0
  run_one prior_shooting 0.0 1.0 0 0
  run_one prior_energy 0.1 1.0 0 0
  run_one prior_energy 0.5 1.0 0 0
  run_one prior_energy 1.0 1.0 0 0
  run_one prior_shooting 0.0 0.8 32 1
}; then
  touch "$RUN_DIR/EVAL_FAILED"
  exit 1
fi
touch "$RUN_DIR/EVAL_COMPLETE"
