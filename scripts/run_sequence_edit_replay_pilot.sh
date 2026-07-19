#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?frozen behavior checkpoint}
seed=${3:-0}
collector_examples=${4:-3000}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

export TMPDIR="/tmp/tj-${RUN_ID:-edit-replay-$$}"
mkdir -p "$TMPDIR"
replay_path="$RUN_DIR/replay.pt"
model_dir="$RUN_DIR/model"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/collect_faithful_token_edit_replay.py" \
  --checkpoint "$checkpoint" --output "$replay_path" \
  --device "${DEVICE:-cuda:0}" --examples "$collector_examples" \
  --candidate-budget 256 --rollout-depth 4

"$python_bin" "${TEXTJEPA_ROOT}/scripts/train.py" \
  +experiment=edit_token_structured_gar_replay_pilot \
  "hydra.run.dir=$model_dir" "run_name=${RUN_ID:-edit-replay-s${seed}}" \
  "seed=$seed" "device=${DEVICE:-cuda:0}" "data.replay_path=$replay_path"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/audit_faithful_token_edits.py" \
  --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
  --examples 256 --corruption-mode mixed --out "$RUN_DIR/metrics.json"
"$python_bin" "${TEXTJEPA_ROOT}/scripts/plan_faithful_token_edits.py" \
  --ckpt "$model_dir/best.pt" --device "${DEVICE:-cuda:0}" \
  --examples 32 --max-candidates 256 --max-steps 32 \
  --out "$RUN_DIR/planning_metrics.json"
