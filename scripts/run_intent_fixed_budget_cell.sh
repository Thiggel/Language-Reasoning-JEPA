#!/usr/bin/env bash
# Fixed-budget iGSM cell: train, shared-menu plan evaluation, and frozen analysis.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || { echo 'RUN_DIR and TEXTJEPA_ROOT required' >&2; exit 2; }
py=${1:?python}; family=${2:?model family}; lr=${3:?lr}
seed=${SEED:-0}; epochs=${EPOCHS:-10}; train_size=${TRAIN_SIZE:-30000}; batch=${BATCH_SIZE:-32}; device=${DEVICE:-cuda:0}
jepa_overrides=()
if [[ -n "${JEPA_OVERRIDES:-}" ]]; then read -r -a jepa_overrides <<< "$JEPA_OVERRIDES"; fi
tmp_dir=${SLURM_TMPDIR:-/tmp/tj-fixed-${SLURM_JOB_ID:-$$}}
mkdir -p "$tmp_dir"; chmod 700 "$tmp_dir"
export TMPDIR="$tmp_dir" TMP="$tmp_dir" TEMP="$tmp_dir"
model_dir="$RUN_DIR/model"; common=("seed=$seed" "device=$device" "train.lr=$lr" "train.epochs=$epochs" "train.batch_size=$batch" "train.num_workers=2" "data.train_size=$train_size" "data.val_size=500" "data.test_size=500" "train.warmup_steps=500" "hydra.run.dir=$model_dir" hydra.output_subdir=null)
case "$family" in
  mlp_jepa|causal_jepa|token_lm)
    BATCH_SIZE="$batch" EPOCHS="$epochs" TRAIN_SIZE="$train_size" SEED="$seed" LM_D_MODEL="${LM_D_MODEL:-272}" RUN_DIR="$RUN_DIR" TEXTJEPA_ROOT="$TEXTJEPA_ROOT" DEVICE="$device" bash "$TEXTJEPA_ROOT/scripts/run_intent_recovery_audit_cell.sh" "$py" igsm "$family" "$lr" "${jepa_overrides[@]}"
    kind=$([ "$family" = token_lm ] && echo token_lm || echo jepa);;
  sentence_lm|sentence_latent_lm)
    exp=paper_sentence_lm; [ "$family" = sentence_latent_lm ] && exp=paper_sentence_latent_lm
    "$py" "$TEXTJEPA_ROOT/scripts/train_sentlm.py" +experiment="$exp" "${common[@]}"
    for slack in 0 2; do "$py" "$TEXTJEPA_ROOT/scripts/plan_sentlm.py" "ckpt=$model_dir/best.pt" "device=$device" split=val n_episodes=200 slack=$slack +score=decoder "out=$RUN_DIR/metrics_slack${slack}.json"; done
    "$py" - "$RUN_DIR" "$family" "$lr" <<'PY'
import json, pathlib, sys
r=pathlib.Path(sys.argv[1]); out={}
for s in (0,2): out[str(s)]=next(iter(json.loads((r/f'metrics_slack{s}.json').read_text()).values()))
(r/'metrics.json').write_text(json.dumps({'family':sys.argv[2],'learning_rate':float(sys.argv[3]),'metrics_by_slack':out},indent=2)+'\n')
PY
    kind=sentence_lm;;
  looped_token_lm)
    loop_values=${EVAL_LOOPS:-"1 2 4 8 16 32"}
    "$py" "$TEXTJEPA_ROOT/scripts/train_lm.py" \
      +experiment=paper_token_lm_looped "${common[@]}" \
      model.d_model="${LM_D_MODEL:-272}" model.n_layers=8 \
      model.n_heads=8 model.ff_mult=4 model.max_len=1024
    for loops in $loop_values; do
      for slack in 0 2; do
        "$py" "$TEXTJEPA_ROOT/scripts/plan_lm.py" \
          "ckpt=$model_dir/best.pt" "device=$device" split=val \
          n_episodes=200 slack="$slack" +eval_loops="$loops" \
          measure_flops=true \
          "out=$RUN_DIR/metrics_loops${loops}_slack${slack}.json"
      done
    done
    kind=token_lm;;
  looped_sentence_lm|looped_sentence_latent_lm)
    loop_values=${EVAL_LOOPS:-"1 2 4 8 16"}
    exp=paper_sentence_lm_looped
    [ "$family" = looped_sentence_latent_lm ] && \
      exp=paper_sentence_latent_lm_looped
    "$py" "$TEXTJEPA_ROOT/scripts/train_sentlm.py" \
      +experiment="$exp" "${common[@]}"
    for loops in $loop_values; do
      for slack in 0 2; do
        "$py" "$TEXTJEPA_ROOT/scripts/plan_sentlm.py" \
          "ckpt=$model_dir/best.pt" "device=$device" split=val \
          n_episodes=200 slack="$slack" +score=decoder \
          +eval_loops="$loops" measure_flops=true \
          "out=$RUN_DIR/metrics_loops${loops}_slack${slack}.json"
      done
    done
    kind=sentence_lm;;
  *) echo "unknown family $family" >&2; exit 2;;
esac
if [[ -n "${loop_values:-}" ]]; then
  EVAL_LOOPS="$loop_values" "$py" - "$RUN_DIR" "$family" "$lr" <<'PY'
import json, os, pathlib, sys, torch
r=pathlib.Path(sys.argv[1]); curves={}
for loops in map(int, os.environ["EVAL_LOOPS"].split()):
    curves[str(loops)]={}
    for slack in (0, 2):
        path=r/f"metrics_loops{loops}_slack{slack}.json"
        curves[str(loops)][str(slack)]=next(iter(json.loads(path.read_text()).values()))
ckpt=torch.load(r/'model/best.pt', map_location='cpu', weights_only=False)
hist_path=r/'model/loop_histogram.json'
hist=json.loads(hist_path.read_text()) if hist_path.exists() else {}
(r/'metrics.json').write_text(json.dumps({
    'family':sys.argv[2], 'learning_rate':float(sys.argv[3]),
    'n_parameters':int(ckpt['n_params']), 'train_loop_histogram':hist,
    'metrics_by_loop_and_slack':curves,
}, indent=2)+'\n')
PY
  for loops in $loop_values; do
    for split in train val; do
      "$py" "$TEXTJEPA_ROOT/scripts/export_intent_representations.py" \
        --checkpoint "$model_dir/best.pt" --kind "$kind" --split "$split" \
        --samples 1024 --device "$device" --eval-loops "$loops" \
        --out "$RUN_DIR/features_loops${loops}_${split}.npz"
    done
    "$py" "$TEXTJEPA_ROOT/scripts/analyze_intent_representations.py" \
      --train "$RUN_DIR/features_loops${loops}_train.npz" \
      --test "$RUN_DIR/features_loops${loops}_val.npz" \
      --out "$RUN_DIR/representation_analysis_loops${loops}.json" --seed "$seed"
  done
  exit 0
fi
for split in train val; do "$py" "$TEXTJEPA_ROOT/scripts/export_intent_representations.py" --checkpoint "$model_dir/best.pt" --kind "$kind" --split "$split" --samples 1024 --device "$device" --out "$RUN_DIR/features_${split}.npz"; done
"$py" "$TEXTJEPA_ROOT/scripts/analyze_intent_representations.py" --train "$RUN_DIR/features_train.npz" --test "$RUN_DIR/features_val.npz" --out "$RUN_DIR/representation_analysis.json" --seed "$seed"
for label in categorical_necessary categorical_operation; do "$py" "$TEXTJEPA_ROOT/scripts/plot_intent_representations.py" --features "$RUN_DIR/features_val.npz" --label "$label" --method pca --seed "$seed" --coordinates-out "$RUN_DIR/${label}_pca.npz" --figure-out "$RUN_DIR/${label}_pca.png"; done
