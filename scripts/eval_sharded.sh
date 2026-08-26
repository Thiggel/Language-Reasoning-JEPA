#!/usr/bin/env bash
# Massively parallel plan_flat eval: shard episodes across many processes
# (env/tokenize work is single-threaded per process) and merge.
#   eval_sharded.sh <ckpt> <out.json> <n_episodes> <shards> <gpu_list> [extra plan_flat flags...]
# e.g. eval_sharded.sh ck.pt out.json 100 24 "0,1,2" --lookahead 8 --endpoints true
set -u
CK=$1; OUT=$2; N=$3; SHARDS=$4; GPUS=$5; shift 5
REPO=/vol/home-vol2/ml/laitenbf/TextJEPA
PY=$REPO/.venv/bin/python
IFS=',' read -ra G <<< "$GPUS"
TMP=$(mktemp -d "${OUT%.json}.shards.XXXX")
export PYTHONPATH=$REPO/src PYTHONUNBUFFERED=1 XDG_CACHE_HOME=/vol/home-vol2/ml/laitenbf/.cache/textjepa
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$REPO"
per=$(( (N + SHARDS - 1) / SHARDS ))
pids=()
for ((s=0; s<SHARDS; s++)); do
  start=$((s * per)); [ $start -ge $N ] && break
  cnt=$per; [ $((start + cnt)) -gt $N ] && cnt=$((N - start))
  gpu=${G[$((s % ${#G[@]}))]}
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/plan_flat.py --ckpt "$CK" --device cuda:0 \
    --n-episodes $cnt --episode-start $start --out "$TMP/shard$s.json" "$@" \
    > "$TMP/shard$s.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ $rc -ne 0 ]; then echo "SHARD FAILURE — logs in $TMP" >&2; exit 1; fi
$PY scripts/merge_plan_shards.py --out "$OUT" "$TMP"/shard*.json && rm -rf "$TMP"
