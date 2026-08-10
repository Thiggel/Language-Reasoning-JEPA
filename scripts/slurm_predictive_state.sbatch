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
[[ $# -gt 0 ]] || {
  echo "a Python script and its arguments are required" >&2
  exit 2
}
: "${TEXTJEPA_PYTHON:?point TEXTJEPA_PYTHON at the cluster shared environment}"
[[ -d "$snapshot/src/textjepa" ]] || {
  echo "snapshot lacks src/textjepa: $snapshot" >&2
  exit 2
}

mkdir -p "$run_dir"
printf '%s\n' RUNNING > "$run_dir/state"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$run_dir/started_at"
export TEXTJEPA_ROOT="$snapshot"
export PYTHONPATH="$snapshot/src"
export RUN_DIR="$run_dir"
export TMPDIR="${SLURM_TMPDIR:-$run_dir/tmp-${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
cd "$snapshot"

"$TEXTJEPA_PYTHON" - "$run_dir" "$snapshot" "$@" <<'PY'
import json
import os
import pathlib
import platform
import socket
import sys

run_dir = pathlib.Path(sys.argv[1])
snapshot = sys.argv[2]
command = sys.argv[3:]
(run_dir / "resolved_config.json").write_text(json.dumps({
    "source_snapshot": snapshot,
    "source_revision": os.environ.get("TEXTJEPA_SOURCE_REVISION", "unknown"),
    "python_command": command,
    "run_id": os.environ.get("RUN_ID", run_dir.name),
}, indent=2) + "\n")
(run_dir / "environment.json").write_text(json.dumps({
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python": sys.version,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "tmpdir": os.environ.get("TMPDIR"),
}, indent=2) + "\n")
PY

finalize() {
  local code=$?
  trap - EXIT
  local final_state
  if [[ "$code" -eq 0 ]]; then final_state=COMPLETED
  elif [[ "$code" -eq 124 ]]; then final_state=TIMEOUT
  elif [[ "$code" -eq 143 || "$code" -eq 130 ]]; then final_state=TERMINATED
  else final_state=FAILED
  fi
  printf '%s\n' "$code" > "$run_dir/exit_code"
  printf '%s\n' "$final_state" > "$run_dir/state"
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$run_dir/finished_at"
  if [[ ! -s "$run_dir/run_summary.json" ]]; then
    "$TEXTJEPA_PYTHON" - "$run_dir" "$final_state" "$code" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
(path / "run_summary.json").write_text(json.dumps({
    "schema_version": 1,
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "scientific_validity": "not_assessed" if sys.argv[2] == "COMPLETED" else "not_admitted",
    "failure_summary": None if sys.argv[2] == "COMPLETED" else "see stderr.log and stdout.log",
}, indent=2) + "\n")
PY
  fi
  exit "$code"
}
trap finalize EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

set +e
"$TEXTJEPA_PYTHON" "$@" >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
code=$?
exit "$code"
