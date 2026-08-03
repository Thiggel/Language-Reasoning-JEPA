#!/usr/bin/env bash
# Train one missing RQ3 control and retain bounded learning-curve snapshots.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo 'RUN_DIR and TEXTJEPA_ROOT required' >&2
  exit 2
}
py=${1:?python executable}
lr=${2:?learning rate}
mode=${3:?distance or direct}
case "$mode" in distance|direct) ;; *) echo "bad mode: $mode" >&2; exit 2;; esac

export PLANNER_ENERGY=$([[ "$mode" == distance ]] && echo oracle_goal || echo value)
snapshot_every=${SNAPSHOT_EVERY_STEPS:-3000}
base_overrides="${JEPA_OVERRIDES:-} model.geo_rank_score_mode=$mode model.value_detach=false train.retain_checkpoint_every_steps=$snapshot_every"
export JEPA_OVERRIDES="$base_overrides"

bash "$TEXTJEPA_ROOT/scripts/run_intent_fixed_budget_cell.sh" \
  "$py" mlp_jepa "$lr"
bash "$TEXTJEPA_ROOT/scripts/run_intent_decision_audit_cell.sh" \
  "$py" jepa "$RUN_DIR/model/best.pt"

# Learning-curve points make the claimed correlation falsifiable across
# optimization time, not only across final seeds.  They are bounded (normally
# three snapshots) and explicitly preserve the geometry-only oracle label.
for checkpoint in "$RUN_DIR"/model/step-*.pt; do
  [[ -e "$checkpoint" ]] || continue
  tag=$(basename "$checkpoint" .pt)
  out_dir="$RUN_DIR/checkpoints/$tag"
  mkdir -p "$out_dir"
  "$py" "$TEXTJEPA_ROOT/scripts/plan.py" \
    "ckpt=$checkpoint" "device=${DEVICE:-cuda:0}" split=val \
    n_episodes="${SNAPSHOT_EPISODES:-200}" slack=0 lookahead=1 \
    "energy=$PLANNER_ENERGY" "out=$out_dir/metrics_slack0.json"
  "$py" "$TEXTJEPA_ROOT/scripts/audit_intent_gar_geometry.py" \
    --checkpoint "$checkpoint" --split val \
    --episodes "${SNAPSHOT_EPISODES:-200}" --seed "${SEED:-1}" \
    --device "${DEVICE:-cuda:0}" --forced-errors \
    --out "$out_dir/val_audit.json" \
    --features-out "$out_dir/val_features.npz"
done
