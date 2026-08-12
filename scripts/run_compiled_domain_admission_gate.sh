#!/usr/bin/env bash
# Full paper admission gate for an already-compiled observed-action domain.
#
# Checks, in order (see projects/intent_phrase/PAPER_EXPERIMENTS.md):
#   1. schema validation, 100% non-oracle catalogue recall of expert actions,
#      exact outcome replay, executor reaches every recorded goal, and
#      disjoint train/validation/test identities,
#   2. the horizon-conditioned Energy losses of the headline recipe are
#      actually live on this domain's compiled rollouts,
#   3. random and oracle reference bounds on the validation split,
#   4. a tiny model overfits a tiny subset, and the action-shuffle falsifier
#      hurts relative to the aligned cell,
#   5. checkpoint invariants: zero dropout, EMA target in eval mode,
#   6. closed-loop evaluation runs with no future action availability, on the
#      shared feasible menu and on the full non-oracle catalogue.
set -euo pipefail

[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
py=${1:?python executable}
domain=${2:?planbench-blocksworld or alfworld-textworld}
data_root=${3:?directory holding train/val/test.jsonl}
seed=${SEED:-0}
device=${DEVICE:-cpu}
tiny_episodes=${TINY_EPISODES:-48}
tiny_epochs=${TINY_EPOCHS:-30}
eval_episodes=${EVAL_EPISODES:-48}
replay_limit=${REPLAY_LIMIT:-0}
chunk_len=${MAX_CHUNK_LEN:-96}
chunks=${MAX_CHUNKS:-64}
# Directory used by the tiny overfit/shuffle cells.  Defaults to the full
# corpus; pointing it at a prepared subset keeps the tiny cells cheap without
# changing which corpus the schema, replay, and bound checks above read.
tiny_data_root=${TINY_DATA_ROOT:-$data_root}
export PYTHONUNBUFFERED=1

case "$domain" in
  planbench-blocksworld) data_name=planbench_blocksworld ;;
  alfworld-textworld) data_name=alfworld ;;
  *) echo "unsupported compiled domain: $domain" >&2; exit 2 ;;
esac

tmp=${SLURM_TMPDIR:-/tmp/tj-gate-${SLURM_JOB_ID:-$$}}
mkdir -p "$tmp" "$RUN_DIR"; chmod 700 "$tmp"
export TMPDIR="$tmp" TMP="$tmp" TEMP="$tmp"

# 1. schema, catalogue recall, exact replay, goal reachability, disjointness
"$py" "$TEXTJEPA_ROOT/scripts/validate_intent_reasoning_data.py" \
  --domain "$domain" \
  --dataset "train=$data_root/train.jsonl" \
  --dataset "val=$data_root/val.jsonl" \
  --dataset "test=$data_root/test.jsonl" \
  --replay-limit "$replay_limit" \
  --out "$RUN_DIR/validation.json"

"$py" "$TEXTJEPA_ROOT/scripts/write_observed_action_data_config.py" \
  --domain "$domain" --root "$data_root" --out "$RUN_DIR/data.yaml"

# 2. the recipe's horizon Energy is live on this domain
"$py" "$TEXTJEPA_ROOT/scripts/check_compiled_domain_horizon_loss.py" \
  --data-config "$RUN_DIR/data.yaml" --episodes 16 \
  --out "$RUN_DIR/horizon_loss.json"

# 3. reference bounds
for reference in random oracle; do
  "$py" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
    --kind "$reference" --data-config "$RUN_DIR/data.yaml" \
    --device "$device" --split val --episodes "$eval_episodes" \
    --excess-actions 0 1 2 4 --seed 7321 \
    --out "$RUN_DIR/${reference}_metrics.json"
done

# 4. tiny overfit and the action-shuffle falsifier
tiny_cell() {
  local name=$1 learning_rate=$2 shuffled=$3
  local model_dir=$RUN_DIR/$name/model
  "$py" "$TEXTJEPA_ROOT/scripts/train.py" \
    +experiment=paper_gar_scoring_screen "data=$data_name" \
    "data.train_path=$tiny_data_root/train.jsonl" \
    "data.val_path=$tiny_data_root/val.jsonl" \
    "data.test_path=$tiny_data_root/test.jsonl" \
    "data.shuffle_actions=$shuffled" \
    data.geo_rank_k=2 data.geo_rank_horizon=8 \
    "data.geo_rank_horizons=[1,2,4,8]" \
    data.geo_rank_candidate_interface=compiled \
    data.dense_geo_anchors=true \
    "data.train_size=$tiny_episodes" "data.val_size=$tiny_episodes" \
    "data.test_size=$tiny_episodes" \
    seed="$seed" device="$device" allow_legacy_predictor=true \
    "train.lr=$learning_rate" "train.epochs=$tiny_epochs" \
    train.batch_size=4 train.num_workers=0 train.warmup_steps=20 \
    train.eval_batches=2 train.log_every=20 \
    model.d_model=64 model.chunk_layers=1 model.chunk_heads=4 \
    model.state_layers=2 model.state_heads=4 model.predictor_layers=2 \
    model.predictor_heads=4 model.ff_mult=2 model.d_action=16 \
    "model.max_chunk_len=$chunk_len" "model.max_chunks=$chunks" \
    model.dropout=0.0 \
    model.geo_rank_score_mode=horizon model.geo_energy_target=distance \
    model.geo_horizon_input=false model.predictor_residual=true \
    model.dense_rollout_depth=0 model.observed_action_ldad=true \
    objective.observed_action_ldad.weight=1.0 \
    objective.geo_rank.weight=0 objective.geo_energy_mse.weight=0 \
    objective.geo_advantage_mse.weight=0.25 \
    objective.geo_horizon_rank.weight=1 \
    objective.geo_horizon_rank.kind=logistic \
    objective.counterfactual_state.weight=1 \
    objective.latent_pred.weight=1 objective.chunk_pred.weight=2 \
    objective.vicreg.weight=1 objective.dense_rollout.weight=0 \
    hydra.run.dir="$model_dir" hydra.output_subdir=null
  # Overfit is measured on the training split; the shared feasible menu is
  # the headline interface and the full catalogue is the stress diagnostic.
  for interface in feasible_menu full; do
    "$py" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
      --kind jepa --checkpoint "$model_dir/last.pt" --device "$device" \
      --split train --episodes "$tiny_episodes" --excess-actions 0 1 2 4 \
      --simulation-depth 2 --jepa-candidate-mode full --beam-width 4 \
      --candidate-interface "$interface" \
      --out "$RUN_DIR/${name}_train_${interface}.json"
  done
}

tiny_cell aligned "${TINY_LR:-0.001}" false
tiny_cell shuffled "${TINY_LR:-0.001}" true

# 5. checkpoint invariants
"$py" "$TEXTJEPA_ROOT/scripts/audit_intent_checkpoint_invariants.py" \
  --checkpoint "$RUN_DIR/aligned/model/last.pt" --device cpu \
  --out "$RUN_DIR/checkpoint_invariants.json"

# 6. closed-loop evaluation on held-out problems
for interface in feasible_menu full; do
  "$py" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
    --kind jepa --checkpoint "$RUN_DIR/aligned/model/last.pt" \
    --device "$device" --split val --episodes "$eval_episodes" \
    --excess-actions 0 1 2 4 --simulation-depth 2 \
    --jepa-candidate-mode full --beam-width 4 \
    --candidate-interface "$interface" \
    --out "$RUN_DIR/aligned_val_${interface}.json"
done

"$py" "$TEXTJEPA_ROOT/scripts/summarize_compiled_domain_gate.py" \
  --run-dir "$RUN_DIR" --domain "$domain" \
  --out "$RUN_DIR/gate_summary.json"
