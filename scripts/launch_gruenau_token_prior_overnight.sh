#!/usr/bin/env bash
# Launch the first 18 cells on GPUs verified free by gruenau-gpus.
set -euo pipefail

ROOT=/vol/home-vol2/ml/laitenbf/TextJEPA
MATRIX="$ROOT/research/hard_text/overnight_token_prior_matrix.tsv"
LOGDIR="$ROOT/runs/overnight_token_prior_logs"
mkdir -p "$LOGDIR"

HOSTS=(7 7 7 10 10 11 11 11 12 12 12 12 12 12 12 12 12 12)
GPUS=(0 2 3 1 2 0 2 3 0 1 2 3 4 5 6 7 8 9)
index=0
tail -n +2 "$MATRIX" | while IFS=$'\t' read -r \
  name weight hidden detach smooth spans dims phases low_dense high_dense purpose
do
  (( index < ${#HOSTS[@]} )) || break
  host="${HOSTS[$index]}"
  gpu="${GPUS[$index]}"
  if [[ -e "$LOGDIR/${name}_train.log" ]]; then
    printf '%s\talready launched; leaving existing writer untouched\n' "$name"
    index=$((index + 1))
    continue
  fi
  printf -v command '%q ' env CUDA_VISIBLE_DEVICES="$gpu" \
    TEXTJEPA_ROOT="$ROOT" "$ROOT/scripts/run_token_prior_overnight_cell.sh" \
    "$name" "$weight" "$hidden" "$detach" "$smooth" "$spans" "$dims" \
    "$phases" "$low_dense" "$high_dense" 6000 3
  remote="cd $(printf %q "$ROOT"); nohup $command > $(printf %q "$LOGDIR/${name}_launcher.log") 2>&1 < /dev/null & echo \$!"
  pid=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
    "laitenbf@gruenau${host}.informatik.hu-berlin.de" "$remote")
  printf '%s\thost=gruenau%s\tgpu=%s\tpid=%s\tpurpose=%s\n' \
    "$name" "$host" "$gpu" "$pid" "$purpose"
  index=$((index + 1))
done
