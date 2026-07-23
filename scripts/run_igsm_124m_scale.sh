#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
kind=${2:?token_lm, sentence_lm, jepa_prior, jepa_no_prior, or jepa_visreg}
name=${3:?unique run name}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

# Python multiprocessing creates Unix-domain sockets below TMPDIR.  Controller
# run paths are intentionally descriptive and can exceed Linux's 108-byte
# socket-path limit, so use a job-unique short local path.
short_tmp="/tmp/tj-${UID}-${SLURM_JOB_ID:-$$}"
mkdir -p "$short_tmp"
export TMPDIR="$short_tmp"

world=${WORLD_GPUS:-${SLURM_GPUS_ON_NODE:-1}}
world=${world%%(*}
if (( 512 % world != 0 )); then
  echo "global batch 512 is not divisible by world size $world" >&2
  exit 2
fi
if (( 64 % world != 0 )); then
  echo "GAR anchor budget 64 is not divisible by world size $world" >&2
  exit 2
fi
model_dir="$RUN_DIR/model"
mkdir -p "$model_dir"

# 1,000 disjoint epochs x 51,200 fresh traces = 51.2M presentations.
common=(
  "hydra.run.dir=$model_dir" "run_name=$name" seed=0
  data.train_size=51200 data.val_size=2048
  'data.n_vars_range=[10,18]' 'data.steps_range=[6,12]'
  train.epochs=1000 train.lr=2e-3 train.weight_decay=0.05
  'train.betas=[0.9,0.98]' train.warmup_steps=1000
  train.precision=bf16 train.num_workers=4 train.eval_batches=16
  train.log_every=100
  'train.checkpoint_examples=[204800,819200,3225600,6400000,12800000,25600000,51200000]'
)
if [[ -n "${RESUME_FROM:-}" ]]; then
  if [[ ! -s "$RESUME_FROM" ]]; then
    echo "resume checkpoint does not exist or is empty: $RESUME_FROM" >&2
    exit 2
  fi
  common+=("train.resume_from=$RESUME_FROM")
fi
launcher=("$python_bin" -m torch.distributed.run --standalone "--nproc_per_node=$world")

case "$kind" in
  token_lm)
    micro_batch=$((512 / world))
    if (( micro_batch > 256 )); then micro_batch=256; fi
    accumulation=$((512 / (micro_batch * world)))
    "${launcher[@]}" "${TEXTJEPA_ROOT}/scripts/train_lm.py" "${common[@]}" \
      train.target_kind=outcome train.rank_weight=0 "train.batch_size=$micro_batch" \
      "train.gradient_accumulation_steps=$accumulation" \
      model.d_model=888 model.n_layers=13 model.n_heads=12 \
      model.ff_mult=4 model.max_len=768
    ;;
  sentence_lm)
    # The matched A100 capacity gate used 8.0 GiB at batch 128. Keep the
    # physical batch at a conservative 256 per rank for both one- and
    # two-GPU runs, and recover global batch 512 by accumulation when needed.
    micro_batch=$((512 / world))
    if (( micro_batch > 256 )); then micro_batch=256; fi
    accumulation=$((512 / (micro_batch * world)))
    "${launcher[@]}" "${TEXTJEPA_ROOT}/scripts/train_sentlm.py" "${common[@]}" \
      train.target_kind=outcome "train.batch_size=$micro_batch" \
      "train.gradient_accumulation_steps=$accumulation" \
      model.d_model=768 model.chunk_layers=2 model.chunk_heads=12 \
      model.state_layers=11 model.state_heads=12 \
      model.dec_layers=3 model.dec_heads=12 model.ff_mult=4 \
      model.max_chunk_len=96 model.max_chunks=64 model.latent_target=false
    ;;
  jepa_prior|jepa_no_prior|jepa_visreg)
    micro_batch=$((512 / world))
    accumulation=$((512 / (micro_batch * world)))
    gar_anchors=$((64 / world))
    use_prior=true
    prior_weight=1
    target_mode=ema
    vicreg_weight=1
    visreg_weight=0
    if [[ "$kind" == jepa_no_prior ]]; then
      use_prior=false
      prior_weight=0
    elif [[ "$kind" == jepa_visreg ]]; then
      target_mode=visreg
      vicreg_weight=0
      visreg_weight=1
    fi
    "${launcher[@]}" "${TEXTJEPA_ROOT}/scripts/train_pooled_sentence_jepa.py" \
      "${common[@]}" "train.batch_size=$micro_batch" train.eval_batch_size=2 \
      "train.gradient_accumulation_steps=$accumulation" model.d_state=768 \
      model.encoder_layers=10 model.pool_heads=12 model.predictor_layers=6 \
      model.n_heads=12 model.ff_mult=4 model.max_len=768 model.d_action=128 \
      model.dense_depth=1 model.dense_checkpoint=false \
      model.pooling_scope=sentence model.use_prefix_decoder=false \
      model.sequence_packing=true model.attention_backend=auto \
      "model.use_token_prior=$use_prior" "objective.token_prior=$prior_weight" \
      "model.target_mode=$target_mode" model.visreg_projections=4096 \
      "objective.vicreg=$vicreg_weight" "objective.visreg=$visreg_weight" \
      "objective.gar_max_anchors=$gar_anchors" objective.dense_discount=1.0
    ;;
  *)
    echo "unknown scale model: $kind" >&2
    exit 2
    ;;
esac

# Compact final metadata. Expensive generation/planning evaluations are run as
# separate checkpoint-qualified jobs so training completion is never delayed.
"$python_bin" - "$model_dir" "$RUN_DIR/metrics.json" "$kind" <<'PY'
import json, pathlib, sys, torch
root, destination, kind = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
checkpoint = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
payload = {
    "kind": kind,
    "parameters": checkpoint.get("n_params"),
    "examples_seen": checkpoint["examples_seen"],
    "optimizer_steps": checkpoint["step"],
    "best_validation": checkpoint.get("metrics"),
    "complete_51_2m": checkpoint["examples_seen"] == 51_200_000,
}
destination.write_text(json.dumps(payload, indent=2) + "\n")
PY
