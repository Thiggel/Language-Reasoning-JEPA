#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_NAME D_MACRO VARIATIONAL"
  exit 2
fi

name=$1
d_macro=$2
variational=$3
log_dir=research/intent_phrase/logs
mkdir -p "$log_dir"
rm -f "$log_dir/${name}.done" "$log_dir/${name}.failed"
trap 'touch "$log_dir/${name}.failed"' ERR

.venv/bin/python scripts/train.py \
  +experiment=intent_hier_screen \
  run_name="$name" \
  model.d_macro="$d_macro" \
  model.macro_variational="$variational"

for slack in 0 2; do
  .venv/bin/python scripts/plan_hierarchical.py \
    ckpt="runs/$name/best.pt" device=cuda:0 n_episodes=100 \
    method=shooting high_horizon=2 n_samples=1024 \
    energy=value slack="$slack"
  .venv/bin/python scripts/plan_hierarchical.py \
    ckpt="runs/$name/best.pt" device=cuda:0 n_episodes=100 \
    method=cem high_horizon=2 n_samples=1200 cem_iters=20 \
    n_elites=10 variance_ema=0.9 energy=value slack="$slack"
done

rm -f "$log_dir/${name}.failed"
touch "$log_dir/${name}.done"
