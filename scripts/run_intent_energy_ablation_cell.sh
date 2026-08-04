#!/usr/bin/env bash
# Train one isolated GAR Energy ablation and evaluate genuine beam depths.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
py=${1:?python executable}
variant=${2:?variant}
lr=${3:-3e-4}
seed=${SEED:-0}; device=${DEVICE:-cuda:0}
epochs=${EPOCHS:-10}; train_size=${TRAIN_SIZE:-30000}; batch=${BATCH_SIZE:-32}
head=transition; target=advantage; source=predicted; mse=0.25; horizon=1
case "$variant" in
  default) ;;
  true_state) source=true ;;
  mse0) mse=0 ;;
  mse01) mse=0.1 ;;
  mse05) mse=0.5 ;;
  mse1) mse=1.0 ;;
  action_distance) target=distance ;;
  state_distance) head=value; target=distance ;;
  h2) horizon=2 ;;
  h4) horizon=4 ;;
  h8) horizon=8 ;;
  h16) horizon=16 ;;
  *) echo "unknown Energy ablation: $variant" >&2; exit 2 ;;
esac

tmp=${SLURM_TMPDIR:-/tmp/tj-energy-${SLURM_JOB_ID:-$$}}
mkdir -p "$tmp"; chmod 700 "$tmp"
export TMPDIR="$tmp" TMP="$tmp" TEMP="$tmp"
model_dir="$RUN_DIR/model"
"$py" "$TEXTJEPA_ROOT/scripts/train.py" +experiment=paper_energy_geometry \
  seed="$seed" device="$device" train.lr="$lr" train.epochs="$epochs" \
  train.batch_size="$batch" train.num_workers=2 data.train_size="$train_size" \
  data.val_size=500 data.test_size=500 train.warmup_steps=500 \
  model.d_model=256 model.chunk_layers=2 model.chunk_heads=4 \
  model.state_layers=4 model.state_heads=8 model.predictor_layers=2 \
  model.ff_mult=4 model.d_action=16 model.max_chunk_len=96 model.max_chunks=96 \
  model.geo_rank_score_mode="$head" model.geo_energy_target="$target" \
  "model.geo_energy_state_source='$source'" \
  objective.geo_energy_mse.weight="$mse" \
  data.geo_rank_horizon="$horizon" data.geo_rank_policy=latent_beam \
  data.geo_rank_beam_width=4 hydra.run.dir="$model_dir" hydra.output_subdir=null

"$py" - "$RUN_DIR/training_complete.json" "$variant" "$seed" "$lr" \
  "$head" "$target" "$source" "$mse" "$horizon" <<'PY'
import json, os, pathlib, sys
p=pathlib.Path(sys.argv[1]); q=p.with_suffix('.tmp')
q.write_text(json.dumps({
    'status':'completed','variant':sys.argv[2],'seed':int(sys.argv[3]),
    'learning_rate':float(sys.argv[4]),'head':sys.argv[5],
    'target':sys.argv[6],'state_source':sys.argv[7],
    'mse_weight':float(sys.argv[8]),'teacher_horizon':int(sys.argv[9]),
}, indent=2)+'\n')
os.replace(q,p)
PY

for depth in 1 4 8 16; do
  oracle=false; [[ "$depth" -gt 1 ]] && oracle=true
  for slack in 0 2; do
    "$py" "$TEXTJEPA_ROOT/scripts/plan.py" ckpt="$model_dir/best.pt" \
      device="$device" split=val n_episodes=200 slack="$slack" \
      lookahead="$depth" max_expand=8 search_algorithm=beam \
      allow_oracle_future_actions="$oracle" \
      out="$RUN_DIR/metrics_depth${depth}_slack${slack}.json"
  done
done
"$py" - "$RUN_DIR" "$variant" <<'PY'
import json, pathlib, sys
r=pathlib.Path(sys.argv[1]); curves={}
for depth in (1,4,8,16):
    curves[str(depth)]={}
    for slack in (0,2):
        data=json.loads((r/f'metrics_depth{depth}_slack{slack}.json').read_text())
        curves[str(depth)][str(slack)]=next(iter(data.values()))
(r/'metrics.json').write_text(json.dumps({
    'variant':sys.argv[2], 'search_algorithm':'global_beam',
    'beam_width':8, 'candidate_protocol':'symbolic-feasible-menu',
    'metrics_by_depth_and_slack':curves,
},indent=2)+'\n')
PY
