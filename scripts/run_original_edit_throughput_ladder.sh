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
effective_batch=${EFFECTIVE_BATCH:-512}
train_size=$((effective_batch * throughput_steps))
extra_jepa=()
if [[ "${EDIT_EMA_VICREG:-0}" == "1" ]]; then
  extra_jepa+=(
    model.sequence_packing=true
    objective.sigreg.weight=0.0
    objective.multiscale_vicreg.weight=0.02
  )
fi
for microbatch in "${microbatches[@]}"; do
  if (( effective_batch % microbatch != 0 )); then
    echo "SKIP microbatch=$microbatch does not divide batch=$effective_batch"
    continue
  fi
  cell="$RUN_DIR/mb${microbatch}"
  mkdir -p "$cell"
  echo "THROUGHPUT_CELL method=$method microbatch=$microbatch"
  if [[ "$method" == "mdlm" ]]; then
    "$python_bin" "$TEXTJEPA_ROOT/scripts/train_edit_mdlm.py" \
      --out "$cell" --device "${DEVICE:-cuda:0}" --seed 0 \
      --train-size "$train_size" --val-size 64 --epochs 1 \
      --max-steps "$throughput_steps" \
      --batch-size "$effective_batch" --microbatch-size "$microbatch" \
      --eval-batches 1 --log-every 1 --num-workers 16 \
      --attention-backend auto >"$cell/stdout.log" 2>"$cell/stderr.log"
  else
    "$python_bin" "$TEXTJEPA_ROOT/scripts/train.py" \
      "+experiment=$method" "hydra.run.dir=$cell/model" \
      "run_name=${RUN_ID}-mb${microbatch}" seed=0 \
      "device=${DEVICE:-cuda:0}" "train.max_steps=$throughput_steps" \
      "data.train_size=$train_size" data.val_size=64 \
      "train.batch_size=$effective_batch" \
      "train.microbatch_size=$microbatch" \
      "train.eval_batch_size=$microbatch" train.num_workers=16 \
      train.eval_batches=1 train.log_every=1 \
      "${extra_jepa[@]}" \
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
"$python_bin" - "$RUN_DIR" "$method" "$effective_batch" \
  "$throughput_steps" "${microbatches[@]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
payload = {
    "method": sys.argv[2],
    "effective_batch": int(sys.argv[3]),
    "optimizer_steps": int(sys.argv[4]),
    "cells": {},
}
for value in sys.argv[5:]:
    cell = root / f"mb{value}"
    if not cell.exists():
        continue
    throughput = memory = None
    csv_path = cell / "model" / "metrics.csv"
    if csv_path.exists():
        import csv
        rows = list(csv.DictReader(csv_path.open()))
        measured = [
            float(row["train/steps_per_s"])
            for row in rows[1:]
            if row.get("train/steps_per_s") not in {None, ""}
        ]
        memories = [
            float(row["train/peak_memory_gib"])
            for row in rows
            if row.get("train/peak_memory_gib") not in {None, ""}
        ]
        if measured:
            measured.sort()
            throughput = measured[len(measured) // 2]
        if memories:
            memory = max(memories)
    elif (cell / "stdout.log").exists():
        for line in (cell / "stdout.log").read_text().splitlines():
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if row.get("optimizer_step", 0) > 0:
                throughput = row.get("updates_per_second", throughput)
            memory = max(memory or 0.0, row.get("peak_memory_gib", 0.0))
    payload["cells"][value] = {
        "state": (cell / "state").read_text().strip(),
        "exit_code": int((cell / "exit_code").read_text()),
        "updates_per_second": throughput,
        "projected_50k_hours": (
            50000 / throughput / 3600 if throughput else None
        ),
        "peak_memory_gib": memory,
    }
(root / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
exit 0
