#!/usr/bin/env bash
set -uo pipefail

python_bin=${1:?python executable}
method=${2:?mdlm or Hydra experiment name}
shift 2
microbatches=("$@")
if [[ ${#microbatches[@]} -eq 0 ]]; then
  microbatches=(8 16 32 64 128)
fi
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
export TMPDIR="/tmp/tj-${RUN_ID:-edit-throughput-$$}"
mkdir -p "$TMPDIR"

# Grünau's shared cache carries the isolated FA4/CUDA-13 stack. Activate it
# only on Hopper; Ampere jobs intentionally stay on their FA2-capable runtime.
fa4_root=${TEXTJEPA_FA4_ROOT:-/vol/home-vol2/ml/laitenbf/.cache/textjepa/flash-attn-4-b23-py311}
gpu_major=$(
  "$python_bin" -c 'import torch; print(torch.cuda.get_device_capability()[0])' \
    2>/dev/null || true
)
if [[ "$gpu_major" =~ ^[0-9]+$ ]] && (( gpu_major >= 9 )) \
    && [[ -d "$fa4_root/flash_attn/cute" ]]; then
  export PYTHONPATH="$fa4_root:$fa4_root/nvidia_cutlass_dsl/python_packages:${PYTHONPATH:-}"
  echo "ATTENTION_ENV flash-attn-4=4.0.0b23 torch=2.13.0+cu130"
else
  echo "ATTENTION_ENV shared-runtime"
fi

overall=0
throughput_steps=${THROUGHPUT_STEPS:-8}
train_size=$((512 * throughput_steps))
for microbatch in "${microbatches[@]}"; do
  cell="$RUN_DIR/mb${microbatch}"
  mkdir -p "$cell"
  echo "THROUGHPUT_CELL method=$method microbatch=$microbatch"
  if [[ "$method" == "mdlm" ]]; then
    "$python_bin" "$TEXTJEPA_ROOT/scripts/train_edit_mdlm.py" \
      --out "$cell" --device "${DEVICE:-cuda:0}" --seed 0 \
      --train-size "$train_size" --val-size 64 --epochs 1 \
      --max-steps "$throughput_steps" \
      --batch-size 512 --microbatch-size "$microbatch" \
      --eval-batches 1 --log-every 1 --num-workers 16 \
      --attention-backend auto >"$cell/stdout.log" 2>"$cell/stderr.log"
  else
    "$python_bin" "$TEXTJEPA_ROOT/scripts/train.py" \
      "+experiment=$method" "hydra.run.dir=$cell/model" \
      "run_name=${RUN_ID}-mb${microbatch}" seed=0 \
      "device=${DEVICE:-cuda:0}" "train.max_steps=$throughput_steps" \
      "data.train_size=$train_size" data.val_size=64 \
      "train.microbatch_size=$microbatch" \
      "train.eval_batch_size=$microbatch" train.num_workers=16 \
      train.eval_batches=1 train.log_every=1 \
      >"$cell/stdout.log" 2>"$cell/stderr.log"
  fi
  rc=$?
  printf '%s\n' "$rc" >"$cell/exit_code"
  if [[ "$rc" -eq 0 ]]; then
    printf '%s\n' COMPLETED >"$cell/state"
  else
    printf '%s\n' FAILED >"$cell/state"
    overall=1
  fi
  tail -n 3 "$cell/stdout.log" || true
  tail -n 3 "$cell/stderr.log" || true
done

# A failed large-microbatch cell is an expected ladder outcome. The job itself
# is successful when every cell left an explicit status for later selection.
"$python_bin" - "$RUN_DIR" "$method" "${microbatches[@]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
payload = {"method": sys.argv[2], "cells": {}}
for value in sys.argv[3:]:
    cell = root / f"mb{value}"
    payload["cells"][value] = {
        "state": (cell / "state").read_text().strip(),
        "exit_code": int((cell / "exit_code").read_text()),
    }
(root / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
exit 0
