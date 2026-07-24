#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
train_run_dir=${2:?training run directory}
method=${3:?mdlm or jepa}
examples=${4:-8}
shift 4 || true
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

deadline=$(( $(date +%s) + 24 * 60 * 60 ))
while true; do
  state=$(tr -d '\n' < "$train_run_dir/state" 2>/dev/null || true)
  if [[ "$state" == COMPLETED ]]; then
    break
  fi
  if [[ "$state" == FAILED ]]; then
    echo "dependency failed: $train_run_dir" >&2
    exit 3
  fi
  if (( $(date +%s) >= deadline )); then
    echo "dependency wait timed out: $train_run_dir" >&2
    exit 4
  fi
  sleep 60
done

"$python_bin" "$TEXTJEPA_ROOT/scripts/eval_original_igsm_lengths.py" \
  --method "$method" --ckpt "$train_run_dir/model/best.pt" \
  --device "${DEVICE:-cuda:0}" --examples-per-cell "$examples" \
  --out "$RUN_DIR/length_metrics.json" "$@"
