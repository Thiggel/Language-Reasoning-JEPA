#!/usr/bin/env bash
# One LM-baseline screen cell: train a language-model row, then evaluate it
# with the SAME slack-curve protocol as the JEPA cells
# (scripts/run_intent_terminal_energy_eval.sh).
#
# Rows (paper contract, "main model rows"):
#   token_lm                  decoder-only token LM, intent-policy target
#   sentence_lm               sentence LM, next-sentence CE only
#   sentence_lm_latent        sentence LM + next-sentence latent MSE
#   token_lm_rec              weight-shared recurrent token LM
#   sentence_lm_rec           weight-shared recurrent sentence LM
#   sentence_lm_latent_rec    weight-shared recurrent sentence LM + latent MSE
#
# All rows are scored by the identical feasible-menu candidate interface used
# by the JEPA planner, from ONE generous-budget run per evaluation setting:
# the policy never reads its budget, so a slack=$MAX_SLACK run yields the exact
# success rate at every smaller slack (success_by_slack) plus per-episode
# excess steps.
#
# The LM rows have NO planning-depth axis (greedy one action at a time, no
# imagined lookahead). The recurrent rows instead have a LOOP axis, evaluated
# at $EVAL_LOOPS. The emitted metrics.json therefore keys its curves by loop
# count and records planning_depth_axis=false, so collectors never mistake a
# loop count for a simulated transition depth.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
py=${1:?python executable}; row=${2:?row}; lr=${3:-3e-4}
seed=${SEED:-0}; device=${DEVICE:-cuda:0}; epochs=${EPOCHS:-10}
train_size=${TRAIN_SIZE:-30000}; batch=${BATCH_SIZE:-32}
episodes=${N_EPISODES:-300}; max_slack=${MAX_SLACK:-4}
width=${WIDTH:-256}; workers=${NUM_WORKERS:-2}
data=${DATA:-igsm}

recurrent=false; latent_target=false
case "$row" in
  token_lm)               kind=token ;;
  token_lm_rec)           kind=token; recurrent=true ;;
  sentence_lm)            kind=sentence ;;
  sentence_lm_rec)        kind=sentence; recurrent=true ;;
  sentence_lm_latent)     kind=sentence; latent_target=true ;;
  sentence_lm_latent_rec) kind=sentence; latent_target=true; recurrent=true ;;
  *) echo "unknown LM row: $row" >&2; exit 2 ;;
esac
# Non-recurrent rows have a single forward pass; the recurrent rows are the
# only ones with a loop axis (paper contract: loops {1,2,4,8,16}).
if [[ "$recurrent" == true ]]; then
  eval_loops=${EVAL_LOOPS:-"1 2 4 8 16"}
else
  eval_loops=${EVAL_LOOPS:-"1"}
fi

model_dir="$RUN_DIR/model"
tmp=${SLURM_TMPDIR:-/tmp/tj-lm-screen-${SLURM_JOB_ID:-$$}}
mkdir -p "$tmp"; chmod 700 "$tmp"
export TMPDIR="$tmp" TMP="$tmp" TEMP="$tmp"

extra=()
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  read -r -a extra_overrides <<< "$EXTRA_OVERRIDES"
  extra+=("${extra_overrides[@]}")
fi

common=(
  "data=$data" seed="$seed" device="$device"
  train.target_kind=intent
  train.lr="$lr" train.epochs="$epochs" train.batch_size="$batch"
  train.num_workers="$workers" train.warmup_steps=500
  train.eval_batches=40
  data.train_size="$train_size" data.val_size=500 data.test_size=500
  model.d_model="$width" "model.recurrent=$recurrent"
  hydra.run.dir="$model_dir" hydra.output_subdir=null
)

if [[ "$kind" == token ]]; then
  "$py" "$TEXTJEPA_ROOT/scripts/train_lm.py" "${common[@]}" "${extra[@]}"
  planner="$TEXTJEPA_ROOT/scripts/plan_lm.py"
  planner_args=()
  metric_key_prefix=lm_intent
else
  "$py" "$TEXTJEPA_ROOT/scripts/train_sentlm.py" "${common[@]}" \
    "model.latent_target=$latent_target" "${extra[@]}"
  planner="$TEXTJEPA_ROOT/scripts/plan_sentlm.py"
  planner_args=()
  metric_key_prefix=sentlm_intent
fi

# The sentence-plus-latent row is ONE trained model with TWO predeclared
# evaluation rules (decoder likelihood and latent distance); both are scored,
# neither is a tuning opportunity.
if [[ "$kind" == sentence && "$latent_target" == true ]]; then
  scores=(decoder latent)
elif [[ "$kind" == sentence ]]; then
  scores=(decoder)
else
  scores=("")
fi

for loops in $eval_loops; do
  loop_arg=()
  if [[ "$recurrent" == true ]]; then
    loop_arg=(eval_loops="$loops")
  fi
  for score in "${scores[@]}"; do
    score_arg=()
    tag="loops${loops}"
    if [[ -n "$score" ]]; then
      score_arg=(score="$score"); tag="$tag-$score"
    fi
    "$py" "$planner" ckpt="$model_dir/best.pt" device="$device" split=val \
      n_episodes="$episodes" slack="$max_slack" slack_curve=true \
      candidate_interface=feasible_menu \
      "${planner_args[@]}" "${loop_arg[@]}" "${score_arg[@]}" \
      out="$RUN_DIR/lm_${tag}_slackcurve.json"
  done
done

"$py" - "$RUN_DIR" "$row" "$model_dir/best.pt" "$episodes" "$max_slack" \
  "$eval_loops" "${scores[*]}" "$recurrent" "$width" "$seed" "$lr" \
  "$latent_target" "$data" <<'PY'
import hashlib, json, pathlib, sys
r = pathlib.Path(sys.argv[1])
episodes, max_slack = int(sys.argv[4]), int(sys.argv[5])
loops_axis = sys.argv[6].split()
scores = [s for s in sys.argv[7].split()] or [""]
curves, by_loops_and_slack = {}, {}
for loops in loops_axis:
    for score in scores:
        tag = f"loops{loops}" + (f"-{score}" if score else "")
        run = next(iter(json.loads(
            (r / f"lm_{tag}_slackcurve.json").read_text()
        ).values()))
        curves[tag] = {
            "success_by_slack": run["success_by_slack"],
            "excess_steps": run["excess_steps"],
        }
        scalars = {k: v for k, v in run.items()
                   if k not in ("success_by_slack", "excess_steps")}
        by_loops_and_slack[tag] = {
            str(s): {**scalars, "slack": s,
                     "success": run["success_by_slack"][str(s)]}
            for s in range(max_slack + 1)
        }
ckpt = pathlib.Path(sys.argv[3])
(r / "metrics.json").write_text(json.dumps({
    "label": sys.argv[2], "checkpoint": str(ckpt),
    "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    "candidate_interface": "feasible_menu",
    "n_episodes": episodes, "max_slack": max_slack,
    # An LM baseline has no imagined-lookahead depth; do not plot it against
    # the JEPA depth axis (use a flat reference line).
    "planning_depth_axis": False,
    "recurrent": sys.argv[8].lower() == "true",
    "eval_loops": loops_axis, "evaluation_rules": scores,
    "d_model": int(sys.argv[9]), "seed": int(sys.argv[10]),
    "learning_rate": float(sys.argv[11]),
    "latent_target": sys.argv[12].lower() == "true",
    "dataset": sys.argv[13],
    "metrics_by_setting_and_slack": by_loops_and_slack,
    "slack_curves": curves,
}, indent=2) + "\n")
(r / "training_complete.json").write_text(json.dumps({
    "status": "completed", "row": sys.argv[2], "seed": int(sys.argv[10]),
    "learning_rate": float(sys.argv[11]), "d_model": int(sys.argv[9]),
    "recurrent": sys.argv[8].lower() == "true",
    "latent_target": sys.argv[12].lower() == "true",
    "dataset": sys.argv[13],
}, indent=2) + "\n")
PY
