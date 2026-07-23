#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

python_bin=${1:?python executable}
family=${2:?model family}
width=${3:?model width}
learning_rate=${4:?learning rate}
device=${DEVICE:-cuda:0}

case "$width" in 128|256|512) ;; *) echo "unsupported width: $width" >&2; exit 2;; esac

case "$family" in
  token_lm|sentence_lm|sentence_latent_lm|looped_token_lm|\
  looped_sentence_lm|looped_sentence_latent_lm|geometry_jepa) ;;
  *) echo "unsupported model family: $family" >&2; exit 2;;
esac

common=(
  "data=igsm_real"
  "device=$device"
  "train.lr=$learning_rate"
  "train.epochs=10"
  "train.batch_size=32"
  "train.num_workers=4"
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

for seed in 0 1 2; do
  seed_dir="$RUN_DIR/seed-$seed"
  model_dir="$seed_dir/model"
  mkdir -p "$seed_dir"
  checkpoint="$model_dir/best.pt"

  if [[ ! -s "$checkpoint" ]]; then
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
          model.max_chunk_len=96 model.max_chunks=96 \
          "hydra.run.dir=$model_dir" hydra.output_subdir=null
        ;;
    esac
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

  if [[ ! -s "$seed_dir/metrics.json" ]]; then
    evaluate "$eval_kind" "$checkpoint" "$seed_dir/metrics.json" full \
      "${eval_args[@]}"
  fi
  if [[ ! -s "$seed_dir/oracle_feasible_metrics.json" ]]; then
    evaluate "$eval_kind" "$checkpoint" \
      "$seed_dir/oracle_feasible_metrics.json" oracle_feasible \
      "${eval_args[@]}"
  fi

  if [[ "$family" == *sentence_latent_lm ]]; then
    latent_args=(--sentence-score latent)
    if [[ "$family" == looped_* ]]; then
      latent_args+=(--eval-loops 4)
    fi
    if [[ ! -s "$seed_dir/latent_score_metrics.json" ]]; then
      evaluate sentence_lm "$checkpoint" \
        "$seed_dir/latent_score_metrics.json" full "${latent_args[@]}"
    fi
    if [[ ! -s "$seed_dir/latent_score_oracle_feasible_metrics.json" ]]; then
      evaluate sentence_lm "$checkpoint" \
        "$seed_dir/latent_score_oracle_feasible_metrics.json" \
        oracle_feasible "${latent_args[@]}"
    fi
  fi
done

"$python_bin" - "$RUN_DIR" "$family" "$width" "$learning_rate" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
members = []
for seed in range(3):
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
    "selection_metric": "mean full-catalogue validation success over excess-action budgets 0,1,2,4",
    "members": members,
    "mean_validation_success_auc": sum(x["validation_success_auc"] for x in members) / len(members),
}
(root / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
