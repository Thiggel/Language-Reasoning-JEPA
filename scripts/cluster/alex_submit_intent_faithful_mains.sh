#!/usr/bin/env bash
# Campaign C: faithful iGSM five-seed main rows on Alex -- PREPARE ONLY.
#
# The learning rate is decided by the local Gruenau LR screen
# (runs/autonomy/intent_phrase/2026-08-11-intent-faithful-screen-v1); it is a
# single parameter here.  Nothing is submitted unless SUBMIT=1 is passed.
#
#   RUN ON ALEX:
#     SNAP_SHA=<sha> bash $SNAP/scripts/cluster/alex_submit_intent_faithful_mains.sh
#     # ... then, once the LR is chosen:
#     SNAP_SHA=<sha> FAITHFUL_LR=3e-4 SUBMIT=1 bash .../alex_submit_intent_faithful_mains.sh
#
# Faithful iGSM generation is CPU-bound (~5 h/epoch at 2 workers), hence
# NUM_WORKERS=16.
set -euo pipefail

SNAP_SHA=${SNAP_SHA:?export SNAP_SHA to the code snapshot sha}
export TEXTJEPA_AUTONOMY_ROOT=${TEXTJEPA_AUTONOMY_ROOT:-/home/atuin/c107fa/c107fa12/TextJEPA-autonomy}
ROOT=$TEXTJEPA_AUTONOMY_ROOT
SNAP=$ROOT/_code/$SNAP_SHA
PY=${TEXTJEPA_PYTHON:-/home/atuin/c107fa/c107fa12/.venv/bin/python}
ROUND=${ROUND:-2026-08-12-intent-faithful-mains-v1}
RUNS=$ROOT/runs/autonomy/intent_phrase/$ROUND
ACCOUNT=${SLURM_ACCOUNT_NAME:-c107fa}
PARTITION=${PARTITION:-a40}
GRES=${GRES:-gpu:a40:1}
TIME_LIMIT=${TIME_LIMIT:-23:55:00}
INNER_TIMEOUT=${INNER_TIMEOUT:-84600}
SEEDS=${SEEDS:-"0 1 2 3 4"}
FAITHFUL_LR=${FAITHFUL_LR:-PENDING_LOCAL_SCREEN}
SUBMIT=${SUBMIT:-0}

[[ -d "$SNAP" ]] || { echo "missing snapshot $SNAP" >&2; exit 2; }
mkdir -p "$RUNS"

FAITHFUL_DATA="data=igsm_real model.max_chunk_len=96 model.max_chunks=96"
LDAD_HEAD="model.observed_action_ldad=true objective.observed_action_ldad.weight=1.0 model.action_codebook_k=256"

for seed in $SEEDS; do
  cell=faith-ldad-mains-s$seed-v1
  run_dir=$RUNS/$cell
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
export EPOCHS=10 TRAIN_SIZE=30000 BATCH_SIZE=32 NUM_WORKERS=16 DEVICE=cuda:0
export N_EPISODES=300 BEAM_WIDTH=8 EVAL_DEPTHS="1 2 4 8 16" MAX_SLACK=4
export SEED=$seed
export EXTRA_OVERRIDES="$FAITHFUL_DATA $LDAD_HEAD"
cd "\$SNAP"
printf 'RUNNING\n' > "\$RUN_DIR/state"
date -u +%Y-%m-%dT%H:%M:%SZ > "\$RUN_DIR/started_at"
"\$SNAP/scripts/cluster/write_cell_environment.sh" "$SNAP_SHA" "$cell" mix4_aux025_nohorizon "$seed" "$FAITHFUL_LR"
timeout --signal=TERM --kill-after=120 $INNER_TIMEOUT \\
  bash scripts/run_intent_horizon_energy_cell.sh $PY mix4_aux025_nohorizon $FAITHFUL_LR \\
  >"\$RUN_DIR/stdout.log" 2>"\$RUN_DIR/stderr.log"
rc=\$?
printf '%s\n' "\$rc" > "\$RUN_DIR/exit_code"
if [ "\$rc" -eq 0 ]; then st=COMPLETED; elif [ "\$rc" -eq 124 ]; then st=TIMEOUT; else st=FAILED; fi
printf '%s\n' "\$st" > "\$RUN_DIR/state"
date -u +%Y-%m-%dT%H:%M:%SZ > "\$RUN_DIR/finished_at"
exit "\$rc"
EOF
  chmod +x "$run_dir/job.sh"
  printf 'PREPARED\n' > "$run_dir/state"
  echo "prepared $cell (lr=$FAITHFUL_LR)"
done

if [[ "$SUBMIT" != 1 ]]; then
  echo "PREPARE ONLY -- nothing submitted. Set FAITHFUL_LR and SUBMIT=1 to launch."
  exit 0
fi
[[ "$FAITHFUL_LR" != PENDING_LOCAL_SCREEN ]] || {
  echo "refusing to submit without FAITHFUL_LR" >&2; exit 2; }
for seed in $SEEDS; do
  cell=faith-ldad-mains-s$seed-v1
  jid=$(sbatch --parsable "$RUNS/$cell/job.sh"); jid=${jid%%;*}
  printf 'PENDING\n' > "$RUNS/$cell/state"
  printf '%s\n' "$jid" > "$RUNS/$cell/slurm_job_id"
  echo "submitted $cell job=$jid"
done
