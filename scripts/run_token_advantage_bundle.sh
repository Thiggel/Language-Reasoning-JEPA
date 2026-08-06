#!/usr/bin/env bash
set -euo pipefail

python_bin=$1
checkpoint=$2
bundle=$3
run_dir=${RUN_DIR:?RUN_DIR must be supplied by researchctl}
mkdir -p "$run_dir"

base=(--ckpt "$checkpoint" --device cuda:0 --examples 256 --epochs 100)

case "$bundle" in
  primitive)
    "$python_bin" scripts/train_token_geometric_advantage.py "${base[@]}" \
      --scope primitive --proposal codebook --horizon 1 --candidates 256 \
      --loss regression --out "$run_dir/primitive_h1_regression.json"
    "$python_bin" scripts/train_token_geometric_advantage.py "${base[@]}" \
      --scope primitive --proposal codebook --horizon 1 --candidates 256 \
      --loss combined --out "$run_dir/primitive_h1_ranked.json"
    "$python_bin" scripts/train_token_geometric_advantage.py "${base[@]}" \
      --scope primitive --proposal codebook --horizon 4 --candidates 32 \
      --loss combined --out "$run_dir/primitive_h4_ranked.json"
    ;;
  codebook-exact)
    "$python_bin" scripts/train_token_geometric_advantage.py "${base[@]}" \
      --scope macro --proposal codebook --outcome-source teacher --horizon 1 \
      --candidates 16 --loss regression \
      --out "$run_dir/codebook_exact_h1_regression.json"
    "$python_bin" scripts/train_token_geometric_advantage.py "${base[@]}" \
      --scope macro --proposal codebook --outcome-source teacher --horizon 1 \
      --candidates 16 --loss combined \
      --out "$run_dir/codebook_exact_h1_ranked.json"
    ;;
  codebook-model)
    "$python_bin" scripts/train_token_geometric_advantage.py "${base[@]}" \
      --scope macro --proposal codebook --outcome-source predicted --horizon 4 \
      --candidates 16 --continuation-candidates 8 --loss combined \
      --out "$run_dir/codebook_model_h4_ranked.json"
    "$python_bin" scripts/train_token_geometric_advantage.py "${base[@]}" \
      --scope macro --proposal perturb --outcome-source predicted --horizon 1 \
      --candidates 16 --loss combined \
      --out "$run_dir/perturbed_code_h1_ranked.json"
    ;;
  prior)
    "$python_bin" scripts/train_token_geometric_advantage.py "${base[@]}" \
      --scope macro --proposal prior --outcome-source predicted --horizon 1 \
      --candidates 16 --loss combined \
      --out "$run_dir/prior_h1_ranked.json"
    "$python_bin" scripts/train_token_geometric_advantage.py "${base[@]}" \
      --scope macro --proposal prior --outcome-source predicted --horizon 4 \
      --candidates 16 --continuation-candidates 8 --loss combined \
      --out "$run_dir/prior_h4_ranked.json"
    ;;
  *) echo "unknown bundle: $bundle" >&2; exit 2 ;;
esac
