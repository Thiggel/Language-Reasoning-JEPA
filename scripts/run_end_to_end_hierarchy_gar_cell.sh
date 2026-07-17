#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
name=${2:?cell name}
seed=${3:?seed}
low_weight=${4:?low GAR weight}
high_weight=${5:?high GAR weight}
level_weights=${6:?per-level GAR weights}
horizon=${7:?GAR teacher horizon}
detach=${8:?detach predicted states}
k=${9:-2}
continuations=${10:-2}
legacy_value_weight=${11:-0}
distinct_states=${12:-true}
rank_objective=${13:-pairwise}
macro_proposals=${14:-global}
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}
model_dir="$run_dir/model"

"$python_bin" scripts/train_token_hierarchy_v2.py \
  "hydra.run.dir=$model_dir" "run_name=$name" "seed=$seed" \
  data.train_size=2000 data.val_size=256 \
  "data.n_vars_range=[10,18]" "data.steps_range=[6,12]" \
  train.epochs=2 train.batch_size=12 train.num_workers=0 \
  train.eval_batches=8 train.warmup_steps=100 \
  model.max_len=768 model.d_model=256 model.encoder_layers=4 \
  model.predictor_layers=2 model.n_heads=8 model.ff_mult=4 model.d_action=64 \
  "model.level_spans=[4,16,64]" "model.level_dims=[32,16,8]" \
  "model.variational_levels=[false]" "model.phase_augmented_levels=[false]" \
  "model.distinct_level_states=$distinct_states" model.level_state_encoder_layers=2 \
  model.low_dense_depth=2 model.high_dense_depth=2 \
  "objective.high_level_weights=[1,1,1]" \
  "objective.low_value=$legacy_value_weight" \
  "objective.high_value=$legacy_value_weight" \
  "objective.geo_rank_low=$low_weight" "objective.geo_rank_high=$high_weight" \
  "objective.geo_rank_level_weights=$level_weights" \
  "objective.geo_rank_horizon=$horizon" "objective.geo_rank_k=$k" \
  "objective.geo_rank_continuations=$continuations" \
  objective.geo_rank_label_gap=0.001 \
  "objective.geo_rank_objective=$rank_objective" \
  "objective.geo_rank_macro_proposals=$macro_proposals" \
  "objective.geo_rank_detach_prediction=$detach"

ckpt="$model_dir/best.pt"
"$python_bin" scripts/audit_token_hierarchy_drift.py \
  --ckpt "$ckpt" --device cuda:0 --examples 64 --max-horizon 16
"$python_bin" scripts/audit_token_hierarchy_gradients.py \
  --ckpt "$ckpt" --device cuda:0 --batch-size 8
"$python_bin" scripts/audit_token_selection.py \
  --ckpt "$ckpt" --device cuda:0 --examples 64 --positions 128
"$python_bin" scripts/probe_token_hierarchy_v2.py \
  --ckpt "$ckpt" --device cuda:0 --examples 256 --max-points 10000

common=(
  --ckpt "$ckpt" --device cuda:0 --episodes 4 --max-tokens 64
  --high-horizon 2 --flat-horizon 32
  --macro-candidates 256 --macro-iterations 5 --macro-elites 32
  --token-candidates 256 --token-iterations 5 --token-elites 32
  --bank-examples 128 --bank-size 1024 --conditional-bank-k 128
)
"$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
  --flat --out "$run_dir/flat_oracle_cem.json"
"$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
  --support-mode conditional_bank --reachability-refine \
  --reach-topn 16 --reach-budget-scale 0.25 \
  --bank-cache "$run_dir/macro_bank.pt" \
  --out "$run_dir/hierarchical_oracle_cem.json"
"$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
  --support-mode conditional_bank --reachability-refine \
  --reach-topn 16 --reach-budget-scale 0.25 \
  --goal-score learned_value --bank-cache "$run_dir/macro_bank.pt" \
  --out "$run_dir/hierarchical_learned_value_cem.json"
"$python_bin" scripts/plan_token_hierarchy_oracle_cem.py "${common[@]}" \
  --support-mode conditional_bank --reachability-refine \
  --reach-topn 16 --reach-budget-scale 0.25 \
  --goal-score combined --bank-cache "$run_dir/macro_bank.pt" \
  --out "$run_dir/hierarchical_combined_cem.json"

"$python_bin" - "$model_dir" "$run_dir/metrics.json" <<'PY'
import csv, json, pathlib, sys
model, destination = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
result = {}
metrics = model / "metrics.csv"
if metrics.exists():
    rows = list(csv.DictReader(metrics.open()))
    result["last_logged_metrics"] = rows[-1] if rows else {}
for name in ("predictor_drift_curves.json", "gradient_diagnostics.json", "token_selection_audit.json", "representation_probes.json"):
    path = model / name
    if path.exists():
        result[path.stem] = json.loads(path.read_text())
for name in ("flat_oracle_cem.json", "hierarchical_oracle_cem.json", "hierarchical_learned_value_cem.json", "hierarchical_combined_cem.json"):
    path = destination.parent / name
    if path.exists():
        result[path.stem] = json.loads(path.read_text())
destination.write_text(json.dumps(result, indent=2) + "\n")
PY
