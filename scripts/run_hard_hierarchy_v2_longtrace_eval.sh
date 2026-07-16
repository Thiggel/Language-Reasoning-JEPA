#!/usr/bin/env bash
# Tiny-N but reachable-horizon check for exploratory hard hierarchy cells.
set -euo pipefail
worker=${1:?worker id}
gpu=${2:?physical gpu id}
log_dir=research/hard_text/logs/hierarchy_v2_longtrace
mkdir -p "$log_dir"

for done in research/hard_text/logs/hierarchy_v2_stage1/*.done; do
  [[ -e "$done" ]] || continue
  name=$(basename "$done" .done)
  # Stable partition independent of shell glob order.
  slot=$(.venv/bin/python - "$name" <<'PY'
import hashlib, sys
print(int(hashlib.sha1(sys.argv[1].encode()).hexdigest(), 16) % 3)
PY
)
  [[ "$slot" == "$worker" ]] || continue
  ckpt="runs/$name/best.pt"
  for mode in flat hierarchy; do
    .venv/bin/python scripts/plan_token_hierarchy_v2.py \
      --hierarchy-ckpt "$ckpt" --proposal-ckpt runs/lm_9m_hard/best.pt \
      --device "cuda:$gpu" --mode "$mode" --episodes 4 \
      --beam 4 --branch 2 --horizon 1 --max-macros 48 \
      --output-tag reachable4 >>"$log_dir/$name.log" 2>&1
  done
  .venv/bin/python scripts/plan_token_hierarchy_v2.py \
    --hierarchy-ckpt "$ckpt" --proposal-ckpt runs/lm_9m_hard/best.pt \
    --device "cuda:$gpu" --mode hierarchy --episodes 4 \
    --beam 4 --branch 2 --horizon 1 --max-macros 48 --oracle-goal \
    --output-tag reachable4 >>"$log_dir/$name.log" 2>&1
  touch "$log_dir/$name.done"
done

