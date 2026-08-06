#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
condition=${2:?experiment config}
seed=${3:?seed}
learning_rate=${4:?learning rate}
epochs=${5:?epochs}
warmup_steps=${6:?warmup steps}
value_weight=${7:?hierarchy value weight}
name="${condition}_lr${learning_rate}_vw${value_weight}_e${epochs}_s${seed}"
model_dir="$RUN_DIR/model"

export TMPDIR="/tmp/tj-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"

"$python_bin" "${TEXTJEPA_ROOT}/scripts/train.py" \
  "+experiment=$condition" \
  "hydra.run.dir=$model_dir" \
  "run_name=$name" \
  "seed=$seed" \
  "train.lr=$learning_rate" \
  "train.epochs=$epochs" \
  "train.warmup_steps=$warmup_steps" \
  "objective.hierarchy_value_steps.weight=$value_weight" \
  "objective.high_dense_value.weight=$value_weight" \
  "device=${DEVICE:-cuda:0}"

PY="$python_bin" bash "${TEXTJEPA_ROOT}/scripts/eval_run.sh" \
  "$model_dir" "${DEVICE:-cuda:0}"

for low_horizon in 1 2; do
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/plan_hierarchical.py" \
    "ckpt=$model_dir/best.pt" \
    "device=${DEVICE:-cuda:0}" \
    "n_episodes=100" \
    "slack=0" \
    "energy=value" \
    "method=shooting" \
    "high_horizon=2" \
    "n_samples=128" \
    "density_weight=0.01" \
    "low_method=discrete" \
    "low_action_source=all_problem" \
    "low_horizon=$low_horizon" \
    "low_max_expand=128" \
    "subgoal_source=model" \
    "out=$model_dir/plan_hier_low${low_horizon}.json"
done

jq -n \
  --arg condition "$condition" \
  --arg learning_rate "$learning_rate" \
  --arg value_weight "$value_weight" \
  --argjson epochs "$epochs" \
  --argjson warmup_steps "$warmup_steps" \
  --slurpfile strict "$model_dir/plan_slack0_look1.json" \
  --slurpfile slack "$model_dir/plan_slack2_look1.json" \
  --slurpfile hier1 "$model_dir/plan_hier_low1.json" \
  --slurpfile hier2 "$model_dir/plan_hier_low2.json" \
  '{condition: $condition, learning_rate: $learning_rate,
    value_weight: $value_weight, epochs: $epochs,
    warmup_steps: $warmup_steps, flat_strict: $strict[0],
    flat_slack2: $slack[0], hierarchy_low1: $hier1[0],
    hierarchy_low2: $hier2[0]}' \
  > "$RUN_DIR/metrics.json"
