#!/usr/bin/env bash
set -euo pipefail
ROOT=/vol/home-vol2/ml/laitenbf/TextJEPA
LOGDIR="$ROOT/runs/overnight_token_prior_logs"
ROWS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)
HOSTS=(10 10 11 11 11 12 12 12 12 12 12 12 12 12)
GPUS=(1 2 0 2 3 0 1 2 4 5 6 7 8 9)
for index in "${!ROWS[@]}"; do
  row="${ROWS[$index]}" host="${HOSTS[$index]}" gpu="${GPUS[$index]}"
  printf -v command '%q ' env CUDA_VISIBLE_DEVICES="$gpu" \
    TEXTJEPA_ROOT="$ROOT" "$ROOT/scripts/run_token_prior_long_cell_v1.sh" "$row"
  remote="cd $(printf %q "$ROOT"); nohup $command > $(printf %q "$LOGDIR/long_r${row}_launcher_v1.log") 2>&1 < /dev/null & echo \$!"
  pid=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
    "laitenbf@gruenau${host}.informatik.hu-berlin.de" "$remote")
  printf 'row=%s\thost=gruenau%s\tgpu=%s\tpid=%s\n' \
    "$row" "$host" "$gpu" "$pid"
done
