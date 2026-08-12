#!/usr/bin/env bash
# Pull a finished (or in-flight) Slurm round back from Alex/Lise into the local
# Gruenau tree, preserving the run-directory layout exactly.
#
#   scripts/cluster/sync_back_round.sh <round-id> [host] [remote-root]
#
# Model checkpoints are large; pass CHECKPOINTS=1 to include them.
set -euo pipefail
round=${1:?round id, e.g. 2026-08-12-intent-long-mains-v1}
host=${2:-alex}
remote_root=${3:-/home/atuin/c107fa/c107fa12/TextJEPA-autonomy}
local_root=${LOCAL_ROOT:-/vol/home-vol2/ml/laitenbf/TextJEPA}
dest=$local_root/runs/autonomy/intent_phrase/$round
mkdir -p "$dest"
excludes=(--exclude 'tmp/' --exclude '*.tmp')
if [[ "${CHECKPOINTS:-0}" != 1 ]]; then
  excludes+=(--exclude 'model/*.pt')
fi
rsync -avz --partial -e 'ssh -o BatchMode=yes' "${excludes[@]}" \
  "$host:$remote_root/runs/autonomy/intent_phrase/$round/" "$dest/"
echo "synced $host:$remote_root/.../$round -> $dest"
