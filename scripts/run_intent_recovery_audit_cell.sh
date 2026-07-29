#!/usr/bin/env bash
# Matched one-seed recovery audit over stylized/faithful iGSM.
set -euo pipefail

[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
python_bin=${1:?python executable}
dataset=${2:?igsm or igsm_real}
family=${3:?mlp_jepa, causal_jepa, or token_lm}
learning_rate=${4:-1e-3}
epochs=${EPOCHS:-10}
train_size=${TRAIN_SIZE:-30000}
seed=${SEED:-0}
device=${DEVICE:-cuda:0}
case "$dataset" in igsm|igsm_real) ;; *) echo "bad dataset: $dataset" >&2; exit 2;; esac
case "$family" in mlp_jepa|causal_jepa|token_lm) ;; *) echo "bad family: $family" >&2; exit 2;; esac
case "$epochs:$train_size" in *[!0-9:]*|:*) echo "bad EPOCHS/TRAIN_SIZE" >&2; exit 2;; esac

tmp=/tmp/tj-${SLURM_JOB_ID:-$$}
mkdir -p "$tmp"; chmod 700 "$tmp"
export TMPDIR=$tmp TMP=$tmp TEMP=$tmp
model_dir="$RUN_DIR/model"
common=("data=$dataset" "seed=$seed" "device=$device" "train.lr=$learning_rate"
  "train.epochs=$epochs" "train.batch_size=32" "train.num_workers=2"
  "data.train_size=$train_size" "data.val_size=500" "data.test_size=500"
  "train.warmup_steps=500" "hydra.run.dir=$model_dir" hydra.output_subdir=null)

case "$family" in
  mlp_jepa)
    "$python_bin" "$TEXTJEPA_ROOT/scripts/train.py" +experiment=paper_recovery_mlp_geometry \
      "${common[@]}" model.d_model=256 model.chunk_layers=2 model.chunk_heads=4 \
      model.state_layers=4 model.state_heads=8 model.predictor_layers=2 \
      model.ff_mult=4 model.d_action=16 model.max_chunk_len=96 model.max_chunks=96
    eval_script=plan.py
    ;;
  causal_jepa)
    "$python_bin" "$TEXTJEPA_ROOT/scripts/train.py" +experiment=paper_recovery_causal_geometry \
      "${common[@]}" model.d_model=256 model.chunk_layers=2 model.chunk_heads=4 \
      model.state_layers=4 model.state_heads=8 model.predictor_layers=2 \
      model.predictor_heads=8 model.ff_mult=4 model.d_action=16 model.max_chunk_len=96 model.max_chunks=96
    eval_script=plan.py
    ;;
  token_lm)
    "$python_bin" "$TEXTJEPA_ROOT/scripts/train_lm.py" +experiment=paper_token_lm_faithful \
      "${common[@]}" model.d_model=256 model.n_layers=8 model.n_heads=8 \
      model.ff_mult=4 model.max_len=1024 model.recurrent=false
    eval_script=plan_lm.py
    ;;
esac

"$python_bin" - "$RUN_DIR/training_complete.json" "$dataset" "$family" "$epochs" "$train_size" "$learning_rate" <<'PY'
import json, os, pathlib, sys
p = pathlib.Path(sys.argv[1]); tmp = p.with_suffix('.tmp')
tmp.write_text(json.dumps({'dataset':sys.argv[2], 'family':sys.argv[3], 'epochs':int(sys.argv[4]), 'train_size':int(sys.argv[5]), 'learning_rate':float(sys.argv[6]), 'status':'completed'}, indent=2)+'\n')
os.replace(tmp, p)
PY

for slack in 0 2; do
  "$python_bin" "$TEXTJEPA_ROOT/scripts/$eval_script" "ckpt=$model_dir/best.pt" \
    "device=$device" split=val n_episodes=200 slack=$slack lookahead=1 \
    "out=$RUN_DIR/metrics_slack${slack}.json"
done
"$python_bin" - "$RUN_DIR" "$dataset" "$family" "$learning_rate" <<'PY'
import json, pathlib, sys
r=pathlib.Path(sys.argv[1]); values={}
for slack in (0,2):
    p=json.loads((r/f'metrics_slack{slack}.json').read_text())
    values[str(slack)]=next(iter(p.values()))
(r/'metrics.json').write_text(json.dumps({'dataset':sys.argv[2], 'family':sys.argv[3], 'learning_rate':float(sys.argv[4]), 'metrics_by_slack':values}, indent=2)+'\n')
PY
