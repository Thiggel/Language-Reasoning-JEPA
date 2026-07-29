#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

job_token=${SLURM_JOB_ID:-$$}
worker_tmp=${SLURM_TMPDIR:-/tmp/tj-$job_token}
if (( ${#worker_tmp} > 60 )); then
  worker_tmp=/tmp/tj-$job_token
fi
mkdir -p "$worker_tmp"
chmod 700 "$worker_tmp"
export TMPDIR=$worker_tmp
export TMP=$worker_tmp
export TEMP=$worker_tmp
echo "multiprocessing_tmp=$worker_tmp"

python_bin=${1:?python executable}
family=${2:?model family}
width=${3:?model width}
learning_rate=${4:?learning rate}
device=${DEVICE:-cuda:0}
epochs=${EPOCHS:-10}
# The geometry experiment defaults to two invalid grounded alternatives.  An
# explicit override is retained solely for the paired validity control below;
# it is ignored by the LM configurations.
invalid_counterfactual_k=${INVALID_COUNTERFACTUAL_K:-2}

case "$width" in 128|256|512) ;; *) echo "unsupported width: $width" >&2; exit 2;; esac
if ! [[ "$epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer: $epochs" >&2
  exit 2
fi

case "$family" in
  token_lm|sentence_lm|sentence_latent_lm|looped_token_lm|\
  looped_sentence_lm|looped_sentence_latent_lm|geometry_jepa) ;;
  *) echo "unsupported model family: $family" >&2; exit 2;;
esac

common=(
  "data=igsm_real"
  "device=$device"
  "train.lr=$learning_rate"
  "train.epochs=$epochs"
  "train.batch_size=32"
  "train.num_workers=0"
  "train.warmup_steps=500"
  "data.train_size=30000"
  "data.val_size=2000"
  "data.test_size=2000"
)

evaluate() {
  local kind=$1 checkpoint=$2 out=$3 interface=$4
  shift 4
  "$python_bin" "${TEXTJEPA_ROOT}/scripts/eval_observed_action.py" \
    --kind "$kind" --checkpoint "$checkpoint" --device "$device" \
    --split val --episodes 500 --excess-actions 0 1 2 4 \
    --candidate-interface "$interface" --seed 7321 "$@" --out "$out"
}

# A checkpoint is written at every epoch by the trainers, so its mere
# existence is not evidence that the requested training budget completed.
# Keep a separate, atomic completion record written only after the trainer
# exits successfully.  This prevents timed-out jobs from being silently
# evaluated and admitted to LR selection as partial runs.
read -r -a seeds <<< "${SEEDS:-0 1 2}"
if (( ${#seeds[@]} == 0 )); then
  echo "SEEDS must contain at least one integer seed" >&2
  exit 2
fi

for seed in "${seeds[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "invalid seed in SEEDS: $seed" >&2
    exit 2
  fi
  seed_dir="$RUN_DIR/seed-$seed"
  model_dir="$seed_dir/model"
  mkdir -p "$seed_dir"
  checkpoint="$model_dir/best.pt"
  completion="$seed_dir/training_complete.json"

  if [[ -s "$checkpoint" && ! -s "$completion" ]]; then
    echo "partial checkpoint at $checkpoint has no training completion record; refusing evaluation or implicit restart" >&2
    echo "submit a new seed-qualified run to restart this seed from scratch" >&2
    exit 75
  fi

  if [[ ! -s "$completion" ]]; then
    case "$family" in
      token_lm|looped_token_lm)
        experiment=paper_token_lm_faithful
        recurrent=false
        if [[ "$family" == looped_token_lm ]]; then
          experiment=paper_token_lm_looped
          recurrent=true
        fi
        "$python_bin" "${TEXTJEPA_ROOT}/scripts/train_lm.py" \
          "+experiment=$experiment" "${common[@]}" "seed=$seed" \
          "model.d_model=$width" model.max_len=1024 \
          "model.recurrent=$recurrent" \
          "hydra.run.dir=$model_dir" hydra.output_subdir=null
        ;;
      sentence_lm|sentence_latent_lm|looped_sentence_lm|\
      looped_sentence_latent_lm)
        experiment=paper_sentence_lm_faithful
        latent=false
        recurrent=false
        if [[ "$family" == sentence_latent_lm ]]; then
          experiment=paper_sentence_latent_lm_faithful
          latent=true
        elif [[ "$family" == looped_sentence_lm ]]; then
          experiment=paper_sentence_lm_looped
          recurrent=true
        elif [[ "$family" == looped_sentence_latent_lm ]]; then
          experiment=paper_sentence_latent_lm_looped
          latent=true
          recurrent=true
        fi
        "$python_bin" "${TEXTJEPA_ROOT}/scripts/train_sentlm.py" \
          "+experiment=$experiment" "${common[@]}" "seed=$seed" \
          "model.d_model=$width" model.max_chunk_len=96 model.max_chunks=96 \
          "model.latent_target=$latent" "model.recurrent=$recurrent" \
          "hydra.run.dir=$model_dir" hydra.output_subdir=null
        ;;
      geometry_jepa)
        "$python_bin" "${TEXTJEPA_ROOT}/scripts/train.py" \
          +experiment=paper_causal_geometry_gar_no_prior \
          "${common[@]}" "seed=$seed" "model.d_model=$width" \
          "data.invalid_counterfactual_k=$invalid_counterfactual_k" \
          data.geo_rank_policy=random \
          model.max_chunk_len=96 model.max_chunks=96 \
          "hydra.run.dir=$model_dir" hydra.output_subdir=null
        ;;
    esac
    "$python_bin" - "$completion" "$seed" "$family" "$width" "$learning_rate" "$invalid_counterfactual_k" "$epochs" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "seed": int(sys.argv[2]),
    "model_family": sys.argv[3],
    "width": int(sys.argv[4]),
    "learning_rate": float(sys.argv[5]),
    "invalid_counterfactual_k": int(sys.argv[6]),
    "epochs": int(sys.argv[7]),
    "status": "completed",
}
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
  fi

  eval_kind=$family
  eval_args=()
  case "$family" in
    token_lm) eval_kind=token_lm ;;
    looped_token_lm)
      eval_kind=token_lm
      eval_args=(--eval-loops 4)
      ;;
    sentence_lm)
      eval_kind=sentence_lm
      eval_args=(--sentence-score decoder)
      ;;
    sentence_latent_lm)
      eval_kind=sentence_lm
      eval_args=(--sentence-score decoder)
      ;;
    looped_sentence_lm)
      eval_kind=sentence_lm
      eval_args=(--sentence-score decoder --eval-loops 4)
      ;;
    looped_sentence_latent_lm)
      eval_kind=sentence_lm
      eval_args=(--sentence-score decoder --eval-loops 4)
      ;;
    geometry_jepa)
      eval_kind=jepa
      eval_args=(--jepa-candidate-mode full --simulation-depth 4 --beam-width 4)
      ;;
  esac

  # The paper interface exposes the current symbolic legal-action menu to
  # every model.  Full-catalogue scoring is retained as an explicit stress
  # diagnostic, never used to select an LR for the reasoning comparison.
  if [[ ! -s "$seed_dir/metrics.json" ]]; then
    evaluate "$eval_kind" "$checkpoint" "$seed_dir/metrics.json" feasible_menu \
      "${eval_args[@]}"
  fi
  if [[ ! -s "$seed_dir/full_catalogue_metrics.json" ]]; then
    evaluate "$eval_kind" "$checkpoint" \
      "$seed_dir/full_catalogue_metrics.json" full \
      "${eval_args[@]}"
  fi

  if [[ "$family" == *sentence_latent_lm ]]; then
    latent_args=(--sentence-score latent)
    if [[ "$family" == looped_* ]]; then
      latent_args+=(--eval-loops 4)
    fi
    if [[ ! -s "$seed_dir/latent_score_metrics.json" ]]; then
      evaluate sentence_lm "$checkpoint" \
        "$seed_dir/latent_score_metrics.json" feasible_menu "${latent_args[@]}"
    fi
    if [[ ! -s "$seed_dir/latent_score_full_catalogue_metrics.json" ]]; then
      evaluate sentence_lm "$checkpoint" \
        "$seed_dir/latent_score_full_catalogue_metrics.json" \
        full "${latent_args[@]}"
    fi
  fi
done

"$python_bin" - "$RUN_DIR" "$family" "$width" "$learning_rate" "${seeds[@]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
members = []
for raw_seed in sys.argv[5:]:
    seed = int(raw_seed)
    path = root / f"seed-{seed}" / "metrics.json"
    payload = json.loads(path.read_text())
    success = payload["metrics_by_excess_actions"]
    auc = sum(float(success[str(budget)]["success"]) for budget in (0, 1, 2, 4)) / 4
    members.append({"seed": seed, "validation_success_auc": auc})
summary = {
    "schema_version": 1,
    "dataset": "igsm_real",
    "model_family": sys.argv[2],
    "width": int(sys.argv[3]),
    "learning_rate": float(sys.argv[4]),
    "selection_metric": "mean shared feasible-menu validation success over excess-action budgets 0,1,2,4",
    "members": members,
    "mean_validation_success_auc": sum(x["validation_success_auc"] for x in members) / len(members),
}
(root / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
