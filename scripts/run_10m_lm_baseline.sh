#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
kind=${2:?token or sentence}
name=${3:?run name}
profile=${4:-pilot}
lr=${5:-3e-4}
budget=${6:-10m}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi
if [[ "$profile" == pilot ]]; then
  train_size=4000; val_size=512; epochs=4; examples=16; warmup=200
elif [[ "$profile" == full ]]; then
  train_size=10000; val_size=2048; epochs=20; examples=32; warmup=1000
else
  echo "profile must be pilot or full" >&2; exit 2
fi
model_dir="$RUN_DIR/model"
common=(
  "hydra.run.dir=$model_dir" "run_name=$name" "train.lr=$lr"
  "data.train_size=$train_size" "data.val_size=$val_size"
  "data.n_vars_range=[10,18]" "data.steps_range=[6,12]"
  "train.epochs=$epochs" "train.warmup_steps=$warmup" train.num_workers=0
)
if [[ "$kind" == token && "$budget" == 10m ]]; then
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/train_lm.py" "${common[@]}" \
    train.target_kind=outcome train.rank_weight=0 train.batch_size=64 \
    model.d_model=320 model.n_layers=8 model.n_heads=8 model.ff_mult=4 model.max_len=768
elif [[ "$kind" == sentence && "$budget" == 10m ]]; then
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/train_sentlm.py" "${common[@]}" \
    train.target_kind=outcome train.batch_size=64 \
    model.d_model=304 model.chunk_layers=2 model.chunk_heads=4 \
    model.state_layers=4 model.state_heads=8 model.dec_layers=2 model.dec_heads=4 \
    model.ff_mult=4 model.max_chunk_len=96 model.max_chunks=64 model.latent_target=false
elif [[ "$kind" == token && "$budget" == matched ]]; then
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/train_lm.py" "${common[@]}" \
    train.target_kind=outcome train.rank_weight=0 train.batch_size=32 \
    model.d_model=520 model.n_layers=13 model.n_heads=8 model.ff_mult=4 model.max_len=768
elif [[ "$kind" == sentence && "$budget" == matched ]]; then
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/train_sentlm.py" "${common[@]}" \
    train.target_kind=outcome train.batch_size=32 \
    model.d_model=496 model.chunk_layers=2 model.chunk_heads=4 \
    model.state_layers=8 model.state_heads=8 model.dec_layers=3 model.dec_heads=4 \
    model.ff_mult=4 model.max_chunk_len=96 model.max_chunks=64 model.latent_target=false
else
  echo "kind must be token or sentence and budget must be 10m or matched" >&2; exit 2
fi
for width in 1 8; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_generative_lm_baseline.py" \
    --kind "$kind" --ckpt "$model_dir/best.pt" --device cuda:0 \
    --examples "$examples" --max-tokens 64 --width "$width" \
    --eval-seed 200003 --output "$model_dir/generation_w${width}.json"
done
"$python_bin" - "$model_dir" "$RUN_DIR/metrics.json" "$kind" "$profile" "$lr" "$budget" <<'PY'
import json, pathlib, sys, torch
root, destination, kind, profile, lr, budget = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6]
checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
payload = {
    "kind": kind, "profile": profile, "lr": lr, "parameter_budget": budget,
    "best_epoch": checkpoint["epoch"], "parameters": checkpoint["n_params"],
    "generation": {
        path.stem: json.loads(path.read_text())
        for path in sorted(root.glob("generation_w*.json"))
    },
}
destination.write_text(json.dumps(payload, indent=2) + "\n")
PY
