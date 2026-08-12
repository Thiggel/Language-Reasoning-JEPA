#!/usr/bin/env bash
# Push an immutable code snapshot (and optionally compiled data) from Gruenau
# to Alex.  Alex has its own filesystem, so nothing under the Gruenau
# /vol/home-vol2 tree is visible there.
#
#   scripts/cluster/sync_snapshot_to_alex.sh <git-ref> [data-dir ...]
#
# Creates $TEXTJEPA_AUTONOMY_ROOT/_code/<full-sha> on Alex from `git archive`
# and rsyncs each extra data directory (given as a path relative to the repo
# root) to the same relative path under $TEXTJEPA_AUTONOMY_ROOT.
set -euo pipefail
ref=${1:?git ref to snapshot}
shift || true
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")/../.." rev-parse --show-toplevel)
sha=$(git -C "$repo" rev-parse "$ref")
host=${ALEX_HOST:-alex}
root=${TEXTJEPA_AUTONOMY_ROOT:-/home/atuin/c107fa/c107fa12/TextJEPA-autonomy}
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

git -C "$repo" archive --format=tar "$sha" | tar -x -C "$stage"
ssh -o BatchMode=yes "$host" "mkdir -p '$root/_code/$sha' '$root/runs/autonomy/intent_phrase'"
rsync -az --delete -e 'ssh -o BatchMode=yes' \
  "$stage"/ "$host:$root/_code/$sha/"

for data_dir in "$@"; do
  ssh -o BatchMode=yes "$host" "mkdir -p '$root/$(dirname "$data_dir")'"
  rsync -az -e 'ssh -o BatchMode=yes' \
    "$repo/$data_dir"/ "$host:$root/$data_dir/"
done

echo "snapshot=$sha root=$root host=$host"
