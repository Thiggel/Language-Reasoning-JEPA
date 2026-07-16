#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RUN_NAME MACRO_SPAN D_MACRO VARIATIONAL"
  exit 2
fi

name=$1
macro_span=$2
d_macro=$3
variational=$4
log_dir=research/hard_text/logs
mkdir -p "$log_dir"
rm -f "$log_dir/${name}.done" "$log_dir/${name}.failed"
trap 'touch "$log_dir/${name}.failed"' ERR

.venv/bin/python scripts/train_token_hierarchy.py \
  +experiment=text_hier_screen \
  run_name="$name" \
  model.macro_span="$macro_span" \
  model.d_macro="$d_macro" \
  model.macro_variational="$variational"

.venv/bin/python scripts/audit_token_hierarchy.py \
  --ckpt "runs/$name/best.pt" --device cuda:0 \
  --examples 512 --prior-samples 256

rm -f "$log_dir/${name}.failed"
touch "$log_dir/${name}.done"
