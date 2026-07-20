#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
checkpoint=${2:?source checkpoint}
episodes=${3:-60}
model_dir="$RUN_DIR/model"

test -f "$checkpoint"
mkdir -p "$model_dir"
sha256sum "$checkpoint" > "$model_dir/source_checkpoint.sha256"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_intent_generalization.py" \
  --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
  --lengths 3 6 9 12 15 18 --slacks 0 1 2 4 \
  --episodes "$episodes" --seed 7321 \
  --out "$model_dir/length_curve_flat.json"

for domain in code prior_noise; do
  for slack in 0 2; do
    "$python_bin" "${TEXTJEPA_ROOT}/scripts/plan_hierarchical.py" \
      "ckpt=$checkpoint" "device=${DEVICE:-cuda:0}" \
      "n_episodes=$episodes" "slack=$slack" energy=value method=cem \
      high_horizon=2 n_samples=1200 cem_iters=20 n_elites=20 \
      "cem_domain=$domain" density_weight=0.01 \
      low_method=discrete low_action_source=all_problem low_horizon=2 \
      low_max_expand=256 subgoal_source=model \
      "out=$model_dir/plan_hier_${domain}_slack${slack}.json"
  done
done

for length in 9 12 15; do
  high=$((length + 8))
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/plan_hierarchical.py" \
    "ckpt=$checkpoint" "device=${DEVICE:-cuda:0}" \
    "n_episodes=$episodes" slack=0 energy=value method=cem \
    high_horizon=2 n_samples=1200 cem_iters=20 n_elites=20 \
    cem_domain=prior_noise density_weight=0.01 \
    low_method=discrete low_action_source=all_problem low_horizon=2 \
    low_max_expand=256 subgoal_source=model \
    "eval_steps_range=[$length,$length]" \
    "eval_n_vars_range=[$length,$high]" \
    "out=$model_dir/plan_hier_prior_noise_len${length}_slack0.json"
done
