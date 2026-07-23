#!/usr/bin/env bash
# Source, compile, replay, bound, and tiny-overfit gate for one paper domain.
set -euo pipefail

if [[ -z "${RUN_DIR:-}" || -z "${TEXTJEPA_ROOT:-}" ]]; then
  echo "RUN_DIR and TEXTJEPA_ROOT must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
gate=${2:?proofwriter or planbench3}
seed=${3:-0}
device=${DEVICE:-cuda:0}
source_root=$RUN_DIR/source
data_root=$RUN_DIR/data
mkdir -p "$source_root" "$data_root"

case "$gate" in
  proofwriter)
    domain=proofwriter
    data_name=proofwriter
    archive=${PROOFWRITER_ARCHIVE:-$source_root/proofwriter-dataset-V2020.12.3.zip}
    if [[ -z "${PROOFWRITER_ARCHIVE:-}" ]]; then
      curl -L --fail --retry 3 -o "$archive" \
        https://allenai.org/data/proofwriter
    fi
    echo "bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26  $archive" \
      | sha256sum --check
    "$python_bin" "$TEXTJEPA_ROOT/scripts/prepare_intent_reasoning_data.py" \
      proofwriter --archive "$archive" --output "$data_root" \
      --train-size 24 --val-size 8 --test-size 8 \
      --teacher-horizon 4 --counterfactual-k 4 --seed 1741
    cat > "$RUN_DIR/source_manifest.json" <<EOF
{"source":"https://allenai.org/data/proofwriter","release":"V2020.12.3","sha256":"bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26","subset":"OWA depth-3 train; OWA depth-5 validation/test"}
EOF
    ;;
  planbench3)
    domain=planbench-blocksworld
    data_name=planbench_blocksworld
    repo=${PLANBENCH_REPO:-$source_root/PlanBench}
    if [[ -z "${PLANBENCH_REPO:-}" ]]; then
      git clone --filter=blob:none --no-checkout \
        https://github.com/harshakokel/PlanBench.git "$repo"
      git -C "$repo" sparse-checkout init --cone
      git -C "$repo" sparse-checkout set \
        plan-bench/instances/blocksworld/generated_basic_3
      git -C "$repo" checkout 1f4d600f4806bedbefebbbe44b168372aaf5060d
    fi
    instances=$repo/plan-bench/instances/blocksworld/generated_basic_3
    "$python_bin" "$TEXTJEPA_ROOT/scripts/prepare_intent_reasoning_data.py" \
      planbench --train-dir "$instances" --val-dir "$instances" \
      --test-dir "$instances" --output "$data_root" \
      --train-size 24 --val-size 8 --test-size 8 \
      --teacher-horizon 4 --counterfactual-k 4 --max-objects 3
    cat > "$RUN_DIR/source_manifest.json" <<EOF
{"source":"https://github.com/harshakokel/PlanBench","commit":"1f4d600f4806bedbefebbbe44b168372aaf5060d","subset":"official generated_basic_3 compiler gate; not full PlanBench admission"}
EOF
    ;;
  *)
    echo "unknown gate: $gate" >&2
    exit 2
    ;;
esac

"$python_bin" "$TEXTJEPA_ROOT/scripts/validate_intent_reasoning_data.py" \
  --domain "$domain" \
  --dataset "train=$data_root/train.jsonl" \
  --dataset "val=$data_root/val.jsonl" \
  --dataset "test=$data_root/test.jsonl" \
  --out "$RUN_DIR/validation.json"
"$python_bin" "$TEXTJEPA_ROOT/scripts/write_observed_action_data_config.py" \
  --domain "$domain" --root "$data_root" --out "$RUN_DIR/data.yaml"

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  exit 0
fi

for reference in random oracle; do
  "$python_bin" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
    --kind "$reference" --data-config "$RUN_DIR/data.yaml" \
    --device "$device" --split val --episodes 8 \
    --excess-actions 0 1 2 4 --seed 7321 \
    --out "$RUN_DIR/${reference}_metrics.json"
done

train_cell() {
  local name=$1 learning_rate=$2 shuffled=$3
  local model_dir=$RUN_DIR/$name/model
  "$python_bin" "$TEXTJEPA_ROOT/scripts/train.py" \
    +experiment=paper_causal_geometry_gar_no_prior "data=$data_name" \
    "data.train_path=$data_root/train.jsonl" \
    "data.val_path=$data_root/val.jsonl" \
    "data.test_path=$data_root/test.jsonl" \
    "data.shuffle_actions=$shuffled" data.geo_rank_k=4 \
    data.geo_rank_horizon=2 data.dense_geo_anchors=true \
    "seed=$seed" "device=$device" "train.lr=$learning_rate" \
    train.epochs=30 train.batch_size=2 train.num_workers=0 \
    train.warmup_steps=20 train.eval_batches=2 train.log_every=20 \
    model.d_model=64 model.chunk_layers=1 model.chunk_heads=4 \
    model.state_layers=2 model.state_heads=4 model.predictor_layers=2 \
    model.predictor_heads=4 model.ff_mult=2 model.d_action=16 \
    model.max_chunk_len=192 model.max_chunks=96 model.dropout=0.0 \
    hydra.run.dir="$model_dir" hydra.output_subdir=null
  for split in train val; do
    "$python_bin" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
      --kind jepa --checkpoint "$model_dir/last.pt" --device "$device" \
      --split "$split" --episodes 24 --excess-actions 0 1 2 4 \
      --simulation-depth 1 --jepa-candidate-mode full --beam-width 4 \
      --out "$RUN_DIR/${name}_${split}_metrics.json"
  done
}

train_cell geometry_lr1e3 0.001 false
train_cell geometry_lr3e3 0.003 false
train_cell shuffled_actions_lr1e3 0.001 true
