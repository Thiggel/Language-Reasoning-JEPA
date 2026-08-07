#!/usr/bin/env bash
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail

snapshot=${1:?snapshot root required}
source_root=${2:?feature/source root required}
checkpoint=${3:?checkpoint required}
run_dir=${4:?run directory required}
method=${5:?method label required}
metric=${6:?metric required}
shift 6

python_bin=/home/atuin/c107fa/c107fa12/.venv/bin/python
deps_root=/home/atuin/c107fa/c107fa12/TextJEPA/_deps
mkdir -p "$run_dir"
printf 'RUNNING\n' > "$run_dir/state"
printf '%s\n' "${SLURM_JOB_ID:-manual}" > "$run_dir/job_id"

finish() {
  code=$?
  printf '%s\n' "$code" > "$run_dir/exit_code"
  if [[ $code -eq 0 ]]; then
    printf 'COMPLETED\n' > "$run_dir/state"
  else
    printf 'FAILED\n' > "$run_dir/state"
  fi
}
trap finish EXIT

if ! command -v module >/dev/null 2>&1; then
  # Alex batch shells are non-login shells on some partitions, so the module
  # function is not initialized automatically.
  source /etc/profile.d/modules.sh
fi
module load cuda/12.8.1
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/atuin/c107fa/c107fa12/.cache/textjepa/huggingface
export PYTHONPATH="$snapshot/src:$deps_root/hf-xet-1.5.2-py312:$deps_root/language-planning-transformers-5.13.0-minimal-py312"
export TMPDIR=${SLURM_TMPDIR:-/tmp/textjepa-${SLURM_JOB_ID:-manual}}
mkdir -p "$TMPDIR"

cd "$snapshot"
"$python_bin" scripts/run_hierarchical_mpc_search_matrix.py \
  --source-root "$source_root" \
  --checkpoint "$checkpoint" \
  --output-root "$run_dir" \
  --method-label "$method" \
  --metric "$metric" \
  "$@"
