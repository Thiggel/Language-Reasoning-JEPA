#!/usr/bin/env bash
set -euo pipefail
ROOT=/vol/home-vol2/ml/laitenbf/TextJEPA
LOGDIR="$ROOT/runs/overnight_token_prior_logs"
# All 13 seed-0 cells plus five seed-1 checks of the central hypotheses.
ROWS=(0 1 2 3 4 5 6 7 8 9 10 11 12 2 3 5 8 11)
SEEDS=(0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1)
HOSTS=(7 7 7 7 10 10 11 11 11 12 12 12 12 12 12 12 12 12)
GPUS=(0 1 2 3 1 2 0 2 3 0 1 2 3 4 5 6 7 8)
for index in "${!ROWS[@]}"; do
  row="${ROWS[$index]}" seed="${SEEDS[$index]}"
  host="${HOSTS[$index]}" gpu="${GPUS[$index]}"
  printf -v command '%q ' env CUDA_VISIBLE_DEVICES="$gpu" \
    TEXTJEPA_ROOT="$ROOT" "$ROOT/scripts/run_token_prior_rollout_cell_v1.sh" \
    "$row" "$seed"
  remote="cd $(printf %q "$ROOT"); nohup $command > $(printf %q "$LOGDIR/rollout_r${row}_s${seed}_launcher_v1.log") 2>&1 < /dev/null & echo \$!"
  pid=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
    "laitenbf@gruenau${host}.informatik.hu-berlin.de" "$remote")
  printf 'row=%s\tseed=%s\thost=gruenau%s\tgpu=%s\tpid=%s\n' \
    "$row" "$seed" "$host" "$gpu" "$pid"
done
