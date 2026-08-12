#!/usr/bin/env bash
# Campaign B: compact ProofWriter learning-rate screen on Alex (NHR@FAU).
#
# ProofWriter is a COMPILED observed-action domain: episodes and teacher
# rollouts are read from JSONL, so the data directory must be synced to Alex
# (scripts/cluster/sync_snapshot_to_alex.sh <ref> data/intent_phrase/proofwriter).
#
#   RUN ON ALEX:
#     SNAP_SHA=<sha> bash $SNAP/scripts/cluster/alex_submit_proofwriter_lr_screen.sh
#
# Cells (seed 0 only, no ablations):
#   pw-ldad-lr{1e-4,3e-4,1e-3,3e-3}-s0-v1     LDAD headline recipe
#   pw-tok-lm-lr{...}-s0-v1                   token LM baseline
#   pw-sent-lm-lr{...}-s0-v1                  sentence LM baseline
set -euo pipefail

SNAP_SHA=${SNAP_SHA:?export SNAP_SHA to the code snapshot sha}
export TEXTJEPA_AUTONOMY_ROOT=${TEXTJEPA_AUTONOMY_ROOT:-/home/atuin/c107fa/c107fa12/TextJEPA-autonomy}
ROOT=$TEXTJEPA_AUTONOMY_ROOT
SNAP=$ROOT/_code/$SNAP_SHA
PY=${TEXTJEPA_PYTHON:-/home/atuin/c107fa/c107fa12/.venv/bin/python}
DATA_ROOT=${DATA_ROOT:-$ROOT/data/intent_phrase/proofwriter}
ROUND=${ROUND:-2026-08-12-intent-proofwriter-lr-screen-v1}
RUNS=$ROOT/runs/autonomy/intent_phrase/$ROUND
ACCOUNT=${SLURM_ACCOUNT_NAME:-c107fa}
PARTITION=${PARTITION:-a40}
GRES=${GRES:-gpu:a40:1}
TIME_LIMIT=${TIME_LIMIT:-23:55:00}
INNER_TIMEOUT=${INNER_TIMEOUT:-84600}
LRS=${LRS:-"1e-4 3e-4 1e-3 3e-3"}
SEED=${SEED:-0}
# Compiled data is a FIXED set of episodes (no fresh-per-epoch sampling), so
# the epoch budget follows the established compiled-domain cells, identically
# for the JEPA row and the LM rows.
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-16}
N_EPISODES=${N_EPISODES:-300}
DRY_RUN=${DRY_RUN:-0}

[[ -d "$SNAP" ]] || { echo "missing snapshot $SNAP" >&2; exit 2; }
for split in train val test; do
  [[ -s "$DATA_ROOT/$split.jsonl" ]] || {
    echo "missing compiled data $DATA_ROOT/$split.jsonl -- sync it first" >&2
    exit 2; }
done
mkdir -p "$RUNS"
: > "$RUNS/$ROUND.submitted.tsv"

lr_tag() { printf '%s' "${1//./}"; }

emit_cell() {  # emit_cell <cell> <runner-line> <envlines>
  local cell=$1 runner_line=$2 envlines=$3
  local run_dir=$RUNS/$cell
  mkdir -p "$run_dir"
  cat > "$run_dir/job.sh" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$cell
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --gres=$GRES
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=$TIME_LIMIT
#SBATCH --output=$run_dir/slurm-%j.out
#SBATCH --error=$run_dir/slurm-%j.err
set -uo pipefail
export SNAP=$SNAP
export TEXTJEPA_AUTONOMY_ROOT=$ROOT
source "\$SNAP/scripts/cluster/alex_env.sh"
RUN_DIR=$run_dir
export RUN_DIR
$envlines
cd "\$SNAP"
printf 'RUNNING\n' > "\$RUN_DIR/state"
date -u +%Y-%m-%dT%H:%M:%SZ > "\$RUN_DIR/started_at"
"\$SNAP/scripts/cluster/write_cell_environment.sh" "$SNAP_SHA" "$cell" proofwriter "$SEED" "-"
timeout --signal=TERM --kill-after=120 $INNER_TIMEOUT \\
  $runner_line \\
  >"\$RUN_DIR/stdout.log" 2>"\$RUN_DIR/stderr.log"
rc=\$?
printf '%s\n' "\$rc" > "\$RUN_DIR/exit_code"
if [ "\$rc" -eq 0 ]; then st=COMPLETED; elif [ "\$rc" -eq 124 ]; then st=TIMEOUT; else st=FAILED; fi
printf '%s\n' "\$st" > "\$RUN_DIR/state"
date -u +%Y-%m-%dT%H:%M:%SZ > "\$RUN_DIR/finished_at"
exit "\$rc"
EOF
  chmod +x "$run_dir/job.sh"
}

submit() {
  local cell=$1
  if [[ "$DRY_RUN" == 1 ]]; then echo "DRY_RUN would submit $cell"; return; fi
  local jid
  jid=$(sbatch --parsable "$RUNS/$cell/job.sh"); jid=${jid%%;*}
  printf 'PENDING\n' > "$RUNS/$cell/state"
  printf '%s\n' "$jid" > "$RUNS/$cell/slurm_job_id"
  printf '%s\t%s\n' "$cell" "$jid" >> "$RUNS/$ROUND.submitted.tsv"
  echo "submitted $cell job=$jid"
}

jepa_env="export SEED=$SEED EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=8 DEVICE=cuda:0
export N_EPISODES=$N_EPISODES BEAM_WIDTH=8 EVAL_DEPTHS=\"1 2 4 8 16\" WIDTH=256
export MAX_CHUNK_LEN=192 MAX_CHUNKS=96 RANK_K=4"
lm_env="export SEED=$SEED EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=8 DEVICE=cuda:0
export N_EPISODES=$N_EPISODES WIDTH=256 DOMAIN=proofwriter
export MAX_LEN=768 MAX_CHUNK_LEN=96 MAX_CHUNKS=64"

for lr in $LRS; do
  tag=$(lr_tag "$lr")
  emit_cell "pw-ldad-lr$tag-s$SEED-v1" \
    "bash scripts/run_compiled_domain_ldad_cell.sh $PY proofwriter $DATA_ROOT $lr" \
    "$jepa_env"
  emit_cell "pw-tok-lm-lr$tag-s$SEED-v1" \
    "bash scripts/run_compiled_domain_lm_cell.sh $PY token_lm $DATA_ROOT $lr" \
    "$lm_env"
  emit_cell "pw-sent-lm-lr$tag-s$SEED-v1" \
    "bash scripts/run_compiled_domain_lm_cell.sh $PY sentence_lm $DATA_ROOT $lr" \
    "$lm_env"
done

for lr in $LRS; do
  tag=$(lr_tag "$lr")
  submit "pw-ldad-lr$tag-s$SEED-v1"
  submit "pw-tok-lm-lr$tag-s$SEED-v1"
  submit "pw-sent-lm-lr$tag-s$SEED-v1"
done

echo "round=$ROUND runs=$RUNS"
