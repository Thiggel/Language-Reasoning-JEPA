#!/usr/bin/env bash
# Train the frozen LDAD headline recipe on a compiled observed-action domain
# and evaluate it closed loop at the paper simulation depths.
#
# The recipe is the horizon-blind endpoint-Energy variant used for iGSM
# (`mix4_aux025_nohorizon` in scripts/run_intent_horizon_energy_cell.sh) plus
# the latent-decoder action-decodability head, kept numerically identical here.
# Compiled domains differ only in where the data comes from: episodes and
# teacher rollouts are read from JSONL instead of being sampled on the fly.
set -euo pipefail

[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
py=${1:?python executable}
domain=${2:?planbench-blocksworld, alfworld-textworld or proofwriter}
data_root=${3:?directory holding train/val/test.jsonl}
lr=${4:-3e-4}

seed=${SEED:-0}; device=${DEVICE:-cuda:0}; epochs=${EPOCHS:-30}
batch=${BATCH_SIZE:-16}; workers=${NUM_WORKERS:-8}
episodes=${N_EPISODES:-200}; width=${BEAM_WIDTH:-8}
depths=${EVAL_DEPTHS:-"1 2 4 8 16"}
chunk_len=${MAX_CHUNK_LEN:-96}; chunks=${MAX_CHUNKS:-64}
width_model=${WIDTH:-256}
rank_k=${RANK_K:-2}

case "$domain" in
  planbench-blocksworld) data_name=planbench_blocksworld ;;
  alfworld-textworld) data_name=alfworld ;;
  proofwriter) data_name=proofwriter ;;
  *) echo "unsupported compiled domain: $domain" >&2; exit 2 ;;
esac

tmp=${SLURM_TMPDIR:-/tmp/tj-compiled-${SLURM_JOB_ID:-$$}}
mkdir -p "$tmp"; chmod 700 "$tmp"
export TMPDIR="$tmp" TMP="$tmp" TEMP="$tmp"
model_dir="$RUN_DIR/model"

"$py" "$TEXTJEPA_ROOT/scripts/write_observed_action_data_config.py" \
  --domain "$domain" --root "$data_root" --out "$RUN_DIR/data.yaml"

"$py" "$TEXTJEPA_ROOT/scripts/train.py" \
  +experiment=paper_gar_scoring_screen "data=$data_name" \
  "data.train_path=$data_root/train.jsonl" \
  "data.val_path=$data_root/val.jsonl" \
  "data.test_path=$data_root/test.jsonl" \
  seed="$seed" device="$device" allow_legacy_predictor=true \
  train.lr="$lr" train.epochs="$epochs" train.batch_size="$batch" \
  train.num_workers="$workers" train.eval_batches=40 \
  train.warmup_steps=500 \
  data.geo_rank_k="$rank_k" data.geo_rank_horizon=8 \
  "data.geo_rank_horizons=[1,2,4,8]" \
  data.geo_rank_candidate_interface=compiled \
  data.dense_geo_anchors=true \
  model.d_model="$width_model" \
  model.max_chunk_len="$chunk_len" model.max_chunks="$chunks" \
  model.dropout=0.0 \
  model.geo_rank_score_mode=horizon model.geo_energy_target=distance \
  model.geo_horizon_input=false model.predictor_residual=true \
  model.geo_td_auxiliary=none \
  model.geo_horizon_supervise_prefixes=false \
  model.dense_rollout_depth=0 \
  model.observed_action_ldad=true \
  objective.observed_action_ldad.weight=1.0 \
  objective.geo_rank.weight=0 objective.geo_energy_mse.weight=0 \
  objective.geo_advantage_mse.weight=0.25 \
  objective.geo_horizon_rank.weight=1 \
  objective.geo_horizon_rank.kind=logistic \
  objective.td_q.weight=0 objective.expectile_value.weight=0 \
  objective.td_jepa.weight=0 objective.goal_head.weight=0 \
  objective.counterfactual_state.weight=1 \
  objective.latent_pred.weight=1 objective.chunk_pred.weight=2 \
  objective.vicreg.weight=1 objective.dense_rollout.weight=0 \
  hydra.run.dir="$model_dir" hydra.output_subdir=null

"$py" "$TEXTJEPA_ROOT/scripts/audit_intent_checkpoint_invariants.py" \
  --checkpoint "$model_dir/best.pt" --device cpu \
  --out "$RUN_DIR/checkpoint_invariants.json"

# Headline interface: the current symbolic legal-action menu, identical for
# every model row.  The full non-oracle catalogue is an additional
# feasibility stress diagnostic, not a headline reasoning metric.
for interface in feasible_menu full; do
  for depth in $depths; do
    "$py" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
      --kind jepa --checkpoint "$model_dir/best.pt" --device "$device" \
      --split val --episodes "$episodes" --excess-actions 0 1 2 4 \
      --simulation-depth "$depth" --jepa-candidate-mode full \
      --beam-width "$width" --candidate-interface "$interface" \
      --out "$RUN_DIR/eval_${interface}_depth${depth}.json"
  done
done

"$py" - "$RUN_DIR" "$domain" "$seed" "$lr" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
(root / "training_complete.json").write_text(json.dumps({
    "status": "completed",
    "recipe": "ldad_mix4_aux025_nohorizon_compiled",
    "domain": sys.argv[2],
    "seed": int(sys.argv[3]),
    "learning_rate": float(sys.argv[4]),
}, indent=2) + "\n")
PY
