#!/usr/bin/env bash
set -euo pipefail
ROOT=/vol/home-vol2/ml/laitenbf/TextJEPA
LOGDIR="$ROOT/runs/overnight_token_prior_logs"
NAMES=(
  linear_w01 linear_w025 linear_w05 linear_w1 linear_w2 linear_w5
  mlp64_w1 mlp256_w05 mlp256_w1 detach_w05 detach_w1 nophase_w1
  dense1_w1 dense4_w1 span8_64_w1
)
HOSTS=(7 7 7 10 10 11 11 11 12 12 12 12 12 12 12)
GPUS=(0 1 2 1 2 0 2 3 0 1 2 4 5 6 7)

for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}" host="${HOSTS[$index]}" gpu="${GPUS[$index]}"
  printf -v command '%q ' env CUDA_VISIBLE_DEVICES="$gpu" \
    TEXTJEPA_ROOT="$ROOT" "$ROOT/scripts/run_token_prior_eval_cell_v1.sh" "$name" 0
  remote="cd $(printf %q "$ROOT"); nohup $command > $(printf %q "$LOGDIR/${name}_eval_launcher_v1.log") 2>&1 < /dev/null & echo \$!"
  pid=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
    "laitenbf@gruenau${host}.informatik.hu-berlin.de" "$remote")
  printf '%s\thost=gruenau%s\tgpu=%s\tpid=%s\n' "$name" "$host" "$gpu" "$pid"
done
