#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 WORKER_ID GPU_ID"
  exit 2
fi

worker=$1
gpu=$2
# Preserve the paired easy-domain matrix. Claim this GPU as soon as its easy
# worker exits, then start the hard-domain hierarchy screen.
while pgrep -f "[r]un_paper_causal_stage1.sh $worker $gpu" >/dev/null; do
  sleep 30
done

# The first easy condition finished training before the flat-planner evaluator
# bug was repaired. Recover that seed on this worker before handing the GPU to
# the hard-domain queue; never retrain it.
j0="runs/paper_causal_j0_s${worker}"
if [[ -f "$j0/TRAIN_DONE" && ! -f "$j0/EVAL_DONE" ]]; then
  if PY=.venv2/bin/python bash scripts/eval_run.sh "$j0" "cuda:$gpu" \
      >>"runs/paper_causal_logs/paper_causal_j0_s${worker}.log" 2>&1; then
    rm -f "$j0/EVAL_FAILED"
    touch "$j0/EVAL_DONE"
  fi
fi
exec bash scripts/run_hard_hierarchy_v2_stage1.sh "$worker" "$gpu"
