#!/usr/bin/env bash
# Collect a broader exact ALFWorld branch fixture once, then compare how much
# of that fixed coverage is used for direct latent/GAR supervision.
set -euo pipefail

if [[ -z "${RUN_DIR:-}" || -z "${TEXTJEPA_ROOT:-}" ]]; then
  echo "RUN_DIR and TEXTJEPA_ROOT must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?training python}
seed=${2:-0}
epochs=${3:-20}
template=${4:-projects/intent_phrase/data/alfworld_pilot_train_games.jsonl}
shared_root=${TEXTJEPA_SHARED_ROOT:-/vol/home-vol2/ml/laitenbf/TextJEPA}
old_pilot=${ALFWORLD_PILOT_ROOT:-$shared_root/data/intent_phrase/alfworld/pilot_v2}
base=$RUN_DIR

collection=$base/collection
mkdir -p "$collection"
RUN_DIR=$collection bash "$TEXTJEPA_ROOT/scripts/run_alfworld_data_gate.sh" \
  train 8 4 4 8 1800 "$template" 24

pilot=$base/broad_pilot
mkdir -p "$pilot/compiled"
cp "$collection/data/compiled/train.jsonl" "$pilot/compiled/train.jsonl"
cp "$old_pilot/compiled/val.jsonl" "$pilot/compiled/val.jsonl"
cp "$old_pilot/compiled/test.jsonl" "$pilot/compiled/test.jsonl"
cp "$collection/data/manifest.json" "$pilot/manifest.json"

run_cell() {
  local name=$1 learning_rate=$2 coverage=$3
  mkdir -p "$base/$name"
  RUN_DIR=$base/$name \
  ALFWORLD_PILOT_ROOT=$pilot \
    bash "$TEXTJEPA_ROOT/scripts/run_alfworld_no_prior_overfit_gate.sh" \
      "$python_bin" geometry_jepa "$seed" "$learning_rate" "$epochs" \
      1 1.0 "$coverage"
}

# K=2 is the information-matched low-coverage control on the exact same
# recollected factual trajectories. K=12 uses all four admissible and eight
# rejected observed branches. The second K=12 learning rate checks the larger
# loss/candidate interaction.
run_cell k2_lr3e3 0.003 2
run_cell k12_lr3e3 0.003 12
run_cell k12_lr1e3 0.001 12
