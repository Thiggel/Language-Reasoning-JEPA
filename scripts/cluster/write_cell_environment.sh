#!/usr/bin/env bash
# Record per-cell job metadata (environment.json) inside $RUN_DIR, mirroring
# what the Gruenau cells write so synced-back directories are self-describing.
#   write_cell_environment.sh <snapshot-sha> <cell> <arg> <seed> <lr>
set -uo pipefail
: "${RUN_DIR:?}"
cat > "$RUN_DIR/environment.json" <<EOF
{
  "cluster": "${SLURM_CLUSTER_NAME:-unknown}",
  "hostname": "$(hostname)",
  "snapshot": "${1:-}",
  "cell": "${2:-}",
  "cell_argument": "${3:-}",
  "seed": ${4:-0},
  "learning_rate": "${5:-}",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "slurm_partition": "${SLURM_JOB_PARTITION:-}",
  "gpus": "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd';' -)",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
