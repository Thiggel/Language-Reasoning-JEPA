#!/usr/bin/env bash
# Campaign A: long-trace stylized iGSM five-seed main rows on Alex (NHR@FAU).
#
# RUN THIS ON ALEX (a login node), from the immutable snapshot:
#   SNAP_SHA=<sha> bash $SNAP/scripts/cluster/alex_submit_intent_long_mains.sh
#
# It writes one run directory per cell under
#   $TEXTJEPA_AUTONOMY_ROOT/runs/autonomy/intent_phrase/$ROUND/<cell>/
# containing job.sh (the sbatch script itself), state, stdout.log, stderr.log,
# exit_code and the cell's result JSONs -- the same layout the Gruenau cells
# use, so a plain rsync drops the results into the local tree unchanged.
#
# Rows (paper contract, long traces: steps 15-25, n_vars 30-60, leaf_prob 0.1):
#   jepa-ldad-long-s{0..4}         LDAD headline recipe (mix4_aux025_nohorizon
#                                  + observed_action_ldad + action codebook)
#   tdjepa-long-s{0..4}            TD-JEPA baseline (arXiv:2510.00739)
#   goalhead-long-s{0..4}          Takai GoalHead baseline
#   tok-lm-long-s{0..4}            token LM              (lr 3e-3)
#   sent-lm-long-s{0..4}           sentence LM           (lr 3e-4)
#   sent-lm-lat-long-s{0..4}       sentence LM + latent  (lr 3e-4)
#   tok-lm-rec-long-s{0..4}        recurrent token LM        (loop axis)
#   sent-lm-rec-long-s{0..4}       recurrent sentence LM     (loop axis)
#   sent-lm-lat-rec-long-s{0..4}   recurrent sentence LM + latent (loop axis)
# The three recurrent rows have never been trained: seed 0 is submitted first
# as a smoke cell and seeds 1-4 are submitted with --dependency=afterok on it.
#
# NO ablations here.  One learning rate knob per family (LDAD_LR etc.) so the
# pending local LR cross-check is a single variable to flip.
set -euo pipefail

SNAP_SHA=${SNAP_SHA:?export SNAP_SHA to the code snapshot sha}
export TEXTJEPA_AUTONOMY_ROOT=${TEXTJEPA_AUTONOMY_ROOT:-/home/atuin/c107fa/c107fa12/TextJEPA-autonomy}
ROOT=$TEXTJEPA_AUTONOMY_ROOT
SNAP=$ROOT/_code/$SNAP_SHA
PY=${TEXTJEPA_PYTHON:-/home/atuin/c107fa/c107fa12/.venv/bin/python}
ROUND=${ROUND:-2026-08-12-intent-long-mains-v1}
RUNS=$ROOT/runs/autonomy/intent_phrase/$ROUND
ACCOUNT=${SLURM_ACCOUNT_NAME:-c107fa}
PARTITION=${PARTITION:-a40}
GRES=${GRES:-gpu:a40:1}
TIME_LIMIT=${TIME_LIMIT:-23:55:00}
# Inner watchdog fires before Slurm kills the step, so state/exit_code are
# always written (Alex caps single-node jobs at 24 h).
INNER_TIMEOUT=${INNER_TIMEOUT:-84600}
SEEDS=${SEEDS:-"0 1 2 3 4"}
LDAD_LR=${LDAD_LR:-3e-4}
TDJEPA_LR=${TDJEPA_LR:-3e-4}
GOALHEAD_LR=${GOALHEAD_LR:-3e-4}
TOKEN_LM_LR=${TOKEN_LM_LR:-3e-3}
SENT_LM_LR=${SENT_LM_LR:-3e-4}
DRY_RUN=${DRY_RUN:-0}
ONLY=${ONLY:-}

[[ -d "$SNAP" ]] || { echo "missing snapshot $SNAP" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "missing python $PY" >&2; exit 2; }
mkdir -p "$RUNS"

# Long-trace data design (held FIXED across train and every eval band, so
# trace length is the only axis).  Sentence-level models cap chunks; the token
# LM instead caps flat token length.
LONG_DATA="data.steps_range=[15,25] data.n_vars_range=[30,60] data.leaf_prob=0.1 data.strict_steps_range=true data.sample_max_tries=20000"
CHUNK_CAP="model.max_chunks=256"
TOKEN_CAP="model.max_len=4096"
LDAD_HEAD="model.observed_action_ldad=true objective.observed_action_ldad.weight=1.0 model.action_codebook_k=256"

submitted=$ROUND.submitted.tsv
: > "$RUNS/$submitted"

# emit_cell <cell> <runner> <arg> <lr> <seed> <extra-overrides> <env-lines>
emit_cell() {
  local cell=$1 runner=$2 arg=$3 lr=$4 seed=$5 overrides=$6 envlines=$7
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
export SEED=$seed
export EXTRA_OVERRIDES="$overrides"
cd "\$SNAP"
printf 'RUNNING\n' > "\$RUN_DIR/state"
date -u +%Y-%m-%dT%H:%M:%SZ > "\$RUN_DIR/started_at"
"\$SNAP/scripts/cluster/write_cell_environment.sh" "$SNAP_SHA" "$cell" "$arg" "$seed" "$lr"
timeout --signal=TERM --kill-after=120 $INNER_TIMEOUT \\
  bash scripts/$runner $PY $arg $lr \\
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

submit() {  # submit <cell> [extra sbatch args...]
  local cell=$1; shift
  if [[ -n "$ONLY" && "$cell" != *"$ONLY"* ]]; then echo "skip $cell"; return; fi
  if [[ "$DRY_RUN" == 1 ]]; then echo "DRY_RUN would submit $cell $*"; return; fi
  local out jid
  out=$(sbatch --parsable "$@" "$RUNS/$cell/job.sh")
  jid=${out%%;*}
  printf 'PENDING\n' > "$RUNS/$cell/state"
  printf '%s\n' "$jid" > "$RUNS/$cell/slurm_job_id"
  printf '%s\t%s\n' "$cell" "$jid" >> "$RUNS/$submitted"
  echo "submitted $cell job=$jid"
  LAST_JOB_ID=$jid
}

jepa_env='export EPOCHS=10 TRAIN_SIZE=30000 BATCH_SIZE=16 NUM_WORKERS=8 DEVICE=cuda:0
export N_EPISODES=300 BEAM_WIDTH=8 EVAL_DEPTHS="1 2 4 8 16" MAX_SLACK=4'
lm_env='export EPOCHS=10 TRAIN_SIZE=30000 BATCH_SIZE=16 NUM_WORKERS=8 DEVICE=cuda:0
export N_EPISODES=300 MAX_SLACK=4 WIDTH=256'

for seed in $SEEDS; do
  emit_cell "jepa-ldad-long-s$seed-v1" run_intent_horizon_energy_cell.sh \
    mix4_aux025_nohorizon "$LDAD_LR" "$seed" \
    "$LONG_DATA $CHUNK_CAP $LDAD_HEAD" "$jepa_env"
  emit_cell "tdjepa-long-s$seed-v1" run_intent_horizon_energy_cell.sh \
    baseline_td_jepa "$TDJEPA_LR" "$seed" \
    "$LONG_DATA $CHUNK_CAP" "$jepa_env"
  emit_cell "goalhead-long-s$seed-v1" run_intent_horizon_energy_cell.sh \
    baseline_goal_head "$GOALHEAD_LR" "$seed" \
    "$LONG_DATA $CHUNK_CAP" "$jepa_env"
  emit_cell "tok-lm-long-s$seed-v1" run_intent_lm_screen_cell.sh \
    token_lm "$TOKEN_LM_LR" "$seed" "$LONG_DATA $TOKEN_CAP" "$lm_env"
  emit_cell "sent-lm-long-s$seed-v1" run_intent_lm_screen_cell.sh \
    sentence_lm "$SENT_LM_LR" "$seed" "$LONG_DATA $CHUNK_CAP" "$lm_env"
  emit_cell "sent-lm-lat-long-s$seed-v1" run_intent_lm_screen_cell.sh \
    sentence_lm_latent "$SENT_LM_LR" "$seed" "$LONG_DATA $CHUNK_CAP" "$lm_env"
  emit_cell "tok-lm-rec-long-s$seed-v1" run_intent_lm_screen_cell.sh \
    token_lm_rec "$TOKEN_LM_LR" "$seed" "$LONG_DATA $TOKEN_CAP" "$lm_env"
  emit_cell "sent-lm-rec-long-s$seed-v1" run_intent_lm_screen_cell.sh \
    sentence_lm_rec "$SENT_LM_LR" "$seed" "$LONG_DATA $CHUNK_CAP" "$lm_env"
  emit_cell "sent-lm-lat-rec-long-s$seed-v1" run_intent_lm_screen_cell.sh \
    sentence_lm_latent_rec "$SENT_LM_LR" "$seed" "$LONG_DATA $CHUNK_CAP" "$lm_env"
done

# Non-recurrent rows: all five seeds independently.
for seed in $SEEDS; do
  for row in jepa-ldad-long tdjepa-long goalhead-long \
             tok-lm-long sent-lm-long sent-lm-lat-long; do
    submit "$row-s$seed-v1"
  done
done

# Recurrent rows: seed 0 is the smoke cell; seeds 1-4 wait for it to succeed.
for row in tok-lm-rec-long sent-lm-rec-long sent-lm-lat-rec-long; do
  LAST_JOB_ID=
  submit "$row-s0-v1"
  smoke=$LAST_JOB_ID
  for seed in $SEEDS; do
    if [[ "$seed" == 0 ]]; then continue; fi
    if [[ -n "$smoke" ]]; then
      submit "$row-s$seed-v1" --dependency=afterok:"$smoke"
    else
      submit "$row-s$seed-v1"
    fi
  done
done

echo "round=$ROUND runs=$RUNS"
echo "submitted cells:"; cat "$RUNS/$submitted"
