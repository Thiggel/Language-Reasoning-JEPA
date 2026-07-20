#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
horizon=${2:?horizon}
ckpt=${3:?checkpoint path}
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}

"$python_bin" scripts/audit_token_oracle_continuation_teacher.py \
  --ckpt "$ckpt" --device cuda:0 --horizon "$horizon" \
  --examples 16 --positions 16 --alternatives 8 \
  --beam-width 4 --beam-branch 4 --out "$run_dir/metrics.json"
