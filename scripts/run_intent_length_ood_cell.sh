#!/usr/bin/env bash
# Post-hoc exact reasoning-length curve for a frozen stylized-iGSM checkpoint.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo 'RUN_DIR and TEXTJEPA_ROOT required' >&2
  exit 2
}
py=${1:?python}; family=${2:?model family}; checkpoint=${3:?checkpoint}
device=${DEVICE:-cuda:0}; episodes=${N_EPISODES:-100}
lengths=${EVAL_LENGTHS:-"3 5 7 9 10 11"}
slacks=${EVAL_SLACKS:-"0 1 2 4"}
n_vars_mode=${N_VARS_MODE:-fixed12}
mkdir -p "$RUN_DIR/cells"

case "$family" in
  mlp_jepa|causal_jepa) planner=scripts/plan.py; extra=();;
  token_lm) planner=scripts/plan_lm.py; extra=();;
  sentence_lm|sentence_latent_lm)
    planner=scripts/plan_sentlm.py; extra=(+score=decoder);;
  *) echo "unknown family: $family" >&2; exit 2;;
esac

for length in $lengths; do
  case "$n_vars_mode" in
    fixed12) n_vars=12;;
    length_plus3) n_vars=$((length + 3));;
    *) echo "unknown N_VARS_MODE: $n_vars_mode" >&2; exit 2;;
  esac
  for slack in $slacks; do
    "$py" "$TEXTJEPA_ROOT/$planner" \
      "ckpt=$checkpoint" "device=$device" split=test \
      "n_episodes=$episodes" "slack=$slack" \
      "eval_steps_range=[$length,$length]" \
      "eval_n_vars_range=[$n_vars,$n_vars]" eval_leaf_prob=0.35 \
      eval_sample_max_tries=100000 eval_strict_steps_range=true \
      "out=$RUN_DIR/cells/length${length}_slack${slack}.json" \
      "${extra[@]}"
  done
done

EVAL_LENGTHS="$lengths" EVAL_SLACKS="$slacks" \
N_VARS_MODE="$n_vars_mode" "$py" - \
  "$RUN_DIR" "$family" "$checkpoint" "$episodes" <<'PY'
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
lengths = [int(x) for x in os.environ["EVAL_LENGTHS"].split()]
slacks = [int(x) for x in os.environ["EVAL_SLACKS"].split()]
n_vars_mode = os.environ["N_VARS_MODE"]
n_vars_by_length = {
    str(length): (12 if n_vars_mode == "fixed12" else length + 3)
    for length in lengths
}
cells = {}
for length in lengths:
    cells[str(length)] = {}
    for slack in slacks:
        path = root / "cells" / f"length{length}_slack{slack}.json"
        cells[str(length)][str(slack)] = next(iter(json.loads(path.read_text()).values()))
payload = {
    "family": sys.argv[2],
    "checkpoint": sys.argv[3],
    "protocol": {
        "dataset": "stylized_iGSM",
        "checkpoint_training_steps_range": [3, 9],
        "n_vars_mode": n_vars_mode,
        "n_vars_by_length": n_vars_by_length,
        "fixed_irrelevant_variables": (
            3 if n_vars_mode == "length_plus3" else None
        ),
        "fixed_leaf_prob": 0.35,
        "id_exact_lengths": [x for x in lengths if x <= 9],
        "ood_exact_lengths": [x for x in lengths if x > 9],
        "slacks": slacks,
        "episodes_per_cell": int(sys.argv[4]),
        "split_seed_source": "checkpoint test_seed",
        "candidate_interface": "current symbolic feasible-action menu",
        "strict_exact_length_sampling": True,
    },
    "cells": cells,
}
(root / "length_curve.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
