#!/usr/bin/env bash
# Fixed-budget iGSM cell: train, shared-menu plan evaluation, and frozen analysis.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || { echo 'RUN_DIR and TEXTJEPA_ROOT required' >&2; exit 2; }
py=${1:?python}; family=${2:?mlp_jepa|causal_jepa|token_lm|sentence_lm|sentence_latent_lm}; lr=${3:?lr}
seed=${SEED:-0}; epochs=${EPOCHS:-10}; train_size=${TRAIN_SIZE:-30000}; batch=${BATCH_SIZE:-32}; device=${DEVICE:-cuda:0}
tmp_dir=${SLURM_TMPDIR:-/tmp/tj-fixed-${SLURM_JOB_ID:-$$}}
mkdir -p "$tmp_dir"; chmod 700 "$tmp_dir"
export TMPDIR="$tmp_dir" TMP="$tmp_dir" TEMP="$tmp_dir"
model_dir="$RUN_DIR/model"; common=("seed=$seed" "device=$device" "train.lr=$lr" "train.epochs=$epochs" "train.batch_size=$batch" "train.num_workers=2" "data.train_size=$train_size" "data.val_size=500" "data.test_size=500" "train.warmup_steps=500" "hydra.run.dir=$model_dir" hydra.output_subdir=null)
case "$family" in
  mlp_jepa|causal_jepa|token_lm)
    BATCH_SIZE="$batch" EPOCHS="$epochs" TRAIN_SIZE="$train_size" SEED="$seed" LM_D_MODEL="${LM_D_MODEL:-272}" RUN_DIR="$RUN_DIR" TEXTJEPA_ROOT="$TEXTJEPA_ROOT" DEVICE="$device" "$TEXTJEPA_ROOT/scripts/run_intent_recovery_audit_cell.sh" "$py" igsm "$family" "$lr"
    kind=$([ "$family" = token_lm ] && echo token_lm || echo jepa);;
  sentence_lm|sentence_latent_lm)
    exp=paper_sentence_lm; [ "$family" = sentence_latent_lm ] && exp=paper_sentence_latent_lm
    "$py" "$TEXTJEPA_ROOT/scripts/train_sentlm.py" +experiment="$exp" "${common[@]}"
    for slack in 0 2; do "$py" "$TEXTJEPA_ROOT/scripts/plan_sentlm.py" "ckpt=$model_dir/best.pt" "device=$device" split=val n_episodes=200 slack=$slack score=decoder "out=$RUN_DIR/metrics_slack${slack}.json"; done
    "$py" - "$RUN_DIR" "$family" "$lr" <<'PY'
import json, pathlib, sys
r=pathlib.Path(sys.argv[1]); out={}
for s in (0,2): out[str(s)]=next(iter(json.loads((r/f'metrics_slack{s}.json').read_text()).values()))
(r/'metrics.json').write_text(json.dumps({'family':sys.argv[2],'learning_rate':float(sys.argv[3]),'metrics_by_slack':out},indent=2)+'\n')
PY
    kind=sentence_lm;;
  *) echo "unknown family $family" >&2; exit 2;;
esac
for split in train val; do "$py" "$TEXTJEPA_ROOT/scripts/export_intent_representations.py" --checkpoint "$model_dir/best.pt" --kind "$kind" --split "$split" --samples 1024 --device "$device" --out "$RUN_DIR/features_${split}.npz"; done
"$py" "$TEXTJEPA_ROOT/scripts/analyze_intent_representations.py" --train "$RUN_DIR/features_train.npz" --test "$RUN_DIR/features_val.npz" --out "$RUN_DIR/representation_analysis.json" --seed "$seed"
for label in categorical_necessary categorical_operation; do "$py" "$TEXTJEPA_ROOT/scripts/plot_intent_representations.py" --features "$RUN_DIR/features_val.npz" --label "$label" --method pca --seed "$seed" --coordinates-out "$RUN_DIR/${label}_pca.npz" --figure-out "$RUN_DIR/${label}_pca.png"; done
