#!/usr/bin/env bash
#SBATCH --job-name=textjepa-pstate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -uo pipefail

snapshot=${1:?exact archived source snapshot required}
run_dir=${2:?project-qualified run directory required}
shift 2
: "${TEXTJEPA_PYTHON:?point TEXTJEPA_PYTHON at the cluster shared environment}"

mkdir -p "$run_dir"
printf '%s\n' RUNNING > "$run_dir/state"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$run_dir/started_at"
export TEXTJEPA_ROOT="$snapshot"
export PYTHONPATH="$snapshot/src"
export RUN_DIR="$run_dir"
export TMPDIR="${SLURM_TMPDIR:-$run_dir/tmp-${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
cd "$snapshot"

set +e
"$TEXTJEPA_PYTHON" "$@" >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
code=$?
set -e
printf '%s\n' "$code" > "$run_dir/exit_code"
if [[ "$code" -eq 0 ]]; then
  printf '%s\n' COMPLETED > "$run_dir/state"
else
  printf '%s\n' FAILED > "$run_dir/state"
fi
exit "$code"
