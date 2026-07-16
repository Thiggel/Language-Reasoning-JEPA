#!/usr/bin/env bash
# Evaluate a frozen final policy on the disjoint test generator exactly once.
# The explicit confirmation guard prevents accidental test-set use during
# validation-based recipe selection.
set -euo pipefail

if [[ ${FINAL_TEST_CONFIRM:-} != "recipe-frozen" ]]; then
  echo "refusing test evaluation: set FINAL_TEST_CONFIRM=recipe-frozen only after model selection" >&2
  exit 2
fi

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 RUN_DIR {latent|lm|sentlm|sentlm-latent} [DEVICE]" >&2
  exit 2
fi

run=${1%/}
kind=$2
device=${3:-cuda:0}
python=${PY:-.venv2/bin/python}
checkpoint="$run/best.pt"

if [[ ! -f $checkpoint ]]; then
  echo "missing frozen checkpoint: $checkpoint" >&2
  exit 2
fi

case "$kind" in
  latent)
    "$python" scripts/plan.py ckpt="$checkpoint" device="$device" \
      split=test slack=0 lookahead=1
    "$python" scripts/plan.py ckpt="$checkpoint" device="$device" \
      split=test slack=2 lookahead=1
    ;;
  lm)
    "$python" scripts/plan_lm.py ckpt="$checkpoint" device="$device" \
      split=test slack=0
    "$python" scripts/plan_lm.py ckpt="$checkpoint" device="$device" \
      split=test slack=2
    ;;
  sentlm)
    "$python" scripts/plan_sentlm.py ckpt="$checkpoint" device="$device" \
      split=test slack=0 +score=decoder
    "$python" scripts/plan_sentlm.py ckpt="$checkpoint" device="$device" \
      split=test slack=2 +score=decoder
    ;;
  sentlm-latent)
    for score in decoder latent; do
      "$python" scripts/plan_sentlm.py ckpt="$checkpoint" device="$device" \
        split=test slack=0 +score="$score"
      "$python" scripts/plan_sentlm.py ckpt="$checkpoint" device="$device" \
        split=test slack=2 +score="$score"
    done
    ;;
  *)
    echo "unknown policy kind: $kind" >&2
    exit 2
    ;;
esac
