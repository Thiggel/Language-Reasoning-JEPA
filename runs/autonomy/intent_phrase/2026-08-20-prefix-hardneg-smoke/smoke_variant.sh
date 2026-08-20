#!/usr/bin/env bash
ulimit -n 65536
set -uo pipefail
ROOT=/vol/home-vol2/ml/laitenbf/TextJEPA
SNAP=$ROOT/runs/autonomy/_code/9d5dc3be937558b163e50526f6e40df4c1709d82
PY=$ROOT/.venv/bin/python
D=$ROOT/runs/autonomy/intent_phrase/2026-08-20-prefix-hardneg-smoke
NAME=$1; KIND=$2; BIAS=$3
export PYTHONPATH=$SNAP/src XDG_CACHE_HOME=/vol/home-vol2/ml/laitenbf/.cache/textjepa
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SNAP"
"$PY" -u "$SNAP/scripts/train_flat_jepa.py" seed=0 device=cuda:0 \
  train.num_workers=6 train.lr=1e-4 train.max_steps=130 train.log_every=10 \
  train.microbatch_size=4 \
  train.eval_batches=2 train.checkpoint_every_steps=100000 \
  objective.energy_cf_feasibility_rank.weight=16 \
  objective.energy_prefix_rank.weight=4 \
  objective.energy_prefix_rank.cf_kind=$KIND \
  objective.energy_prefix_rank.depth_bias=$BIAS \
  hydra.run.dir="$D/$NAME" > "$D/$NAME.log" 2>&1
echo "$(date -u +%FT%TZ) $NAME rc=$?" >> $D/progress.log
