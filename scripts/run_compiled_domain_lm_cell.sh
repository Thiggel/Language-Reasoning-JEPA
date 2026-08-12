#!/usr/bin/env bash
# One LM-baseline cell on a compiled observed-action domain (ProofWriter,
# PlanBench, ALFWorld): train a language-model row on the compiled JSONL
# episodes, then evaluate it closed loop with the SAME information-matched
# protocol the LDAD cell uses (scripts/eval_observed_action.py), so the LM
# rows and the JEPA row differ only in the model.
#
# Sibling of scripts/run_compiled_domain_ldad_cell.sh (JEPA/LDAD row) and of
# scripts/run_intent_lm_screen_cell.sh (the on-the-fly iGSM LM screen). The
# iGSM screen cannot be reused here because its evaluators
# (scripts/plan_lm.py / scripts/plan_sentlm.py) sample iGSM problems, whereas
# compiled domains must replay recorded episodes.
#
# Rows:
#   token_lm     decoder-only token LM, intent-policy target
#   sentence_lm  sentence LM, next-sentence CE only
#
# LM rows have no imagined-lookahead depth (greedy one action at a time), so
# metrics.json records planning_depth_axis=false. The evaluator reports one
# aggregate per excess-action budget, not a slack curve, so no slack_curve
# field is emitted.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
py=${1:?python executable}
row=${2:?token_lm or sentence_lm}
data_root=${3:?directory holding train/val/test.jsonl}
lr=${4:-3e-4}

seed=${SEED:-0}; device=${DEVICE:-cuda:0}; epochs=${EPOCHS:-10}
batch=${BATCH_SIZE:-32}; workers=${NUM_WORKERS:-2}
episodes=${N_EPISODES:-300}; width_model=${WIDTH:-256}
# null uses every compiled episode in the split.
train_size=${TRAIN_SIZE:-null}
val_size=${VAL_SIZE:-null}
test_size=${TEST_SIZE:-null}
max_len=${MAX_LEN:-768}
chunk_len=${MAX_CHUNK_LEN:-96}; chunks=${MAX_CHUNKS:-64}
excess=${EXCESS_ACTIONS:-"0 1 2 4"}

case "$row" in
  token_lm) kind=token ;;
  sentence_lm) kind=sentence ;;
  *) echo "unsupported compiled-domain LM row: $row" >&2; exit 2 ;;
esac

# The data config group name matches the domain for the compiled domains.
domain=${DOMAIN:-proofwriter}
case "$domain" in
  planbench-blocksworld) data_name=planbench_blocksworld ;;
  alfworld-textworld) data_name=alfworld ;;
  proofwriter) data_name=proofwriter ;;
  *) echo "unsupported compiled domain: $domain" >&2; exit 2 ;;
esac

tmp=${SLURM_TMPDIR:-/tmp/tj-compiled-lm-${SLURM_JOB_ID:-$$}}
mkdir -p "$tmp"; chmod 700 "$tmp"
export TMPDIR="$tmp" TMP="$tmp" TEMP="$tmp"
model_dir="$RUN_DIR/model"

"$py" "$TEXTJEPA_ROOT/scripts/write_observed_action_data_config.py" \
  --domain "$domain" --root "$data_root" --out "$RUN_DIR/data.yaml"

extra=()
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  read -r -a extra_overrides <<< "$EXTRA_OVERRIDES"
  extra+=("${extra_overrides[@]}")
fi

common=(
  "data=$data_name"
  "data.train_path=$data_root/train.jsonl"
  "data.val_path=$data_root/val.jsonl"
  "data.test_path=$data_root/test.jsonl"
  "data.train_size=$train_size" "data.val_size=$val_size"
  "data.test_size=$test_size"
  seed="$seed" device="$device"
  train.target_kind=intent
  train.lr="$lr" train.epochs="$epochs" train.batch_size="$batch"
  train.num_workers="$workers" train.warmup_steps=500
  train.eval_batches=40
  model.d_model="$width_model" model.recurrent=false
  hydra.run.dir="$model_dir" hydra.output_subdir=null
)

if [[ "$kind" == token ]]; then
  "$py" "$TEXTJEPA_ROOT/scripts/train_lm.py" "${common[@]}" \
    "model.max_len=$max_len" "${extra[@]}"
else
  "$py" "$TEXTJEPA_ROOT/scripts/train_sentlm.py" "${common[@]}" \
    model.latent_target=false \
    "model.max_chunk_len=$chunk_len" "model.max_chunks=$chunks" \
    "${extra[@]}"
fi

# Headline interface: the current symbolic legal-action menu, identical for
# every model row (same call the LDAD cell makes for --kind jepa).
"$py" "$TEXTJEPA_ROOT/scripts/eval_observed_action.py" \
  --kind "$row" --checkpoint "$model_dir/best.pt" --device "$device" \
  --split val --episodes "$episodes" --excess-actions $excess \
  --candidate-interface feasible_menu \
  --out "$RUN_DIR/eval_feasible_menu.json"

"$py" - "$RUN_DIR" "$row" "$model_dir/best.pt" "$domain" "$episodes" \
  "$width_model" "$seed" "$lr" "$excess" <<'PY'
import hashlib, json, pathlib, sys
r = pathlib.Path(sys.argv[1])
ckpt = pathlib.Path(sys.argv[3])
run = json.loads((r / "eval_feasible_menu.json").read_text())
(r / "metrics.json").write_text(json.dumps({
    "label": sys.argv[2],
    "row": sys.argv[2],
    "checkpoint": str(ckpt),
    "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    "domain": sys.argv[4],
    "dataset": sys.argv[4],
    "candidate_interface": run["candidate_interface"],
    "n_episodes": int(sys.argv[5]),
    "excess_actions": [int(v) for v in sys.argv[9].split()],
    # An LM baseline has no imagined-lookahead depth; do not plot it against
    # the JEPA depth axis (use a flat reference line).
    "planning_depth_axis": False,
    "recurrent": False,
    "latent_target": False,
    "d_model": int(sys.argv[6]),
    "seed": int(sys.argv[7]),
    "learning_rate": float(sys.argv[8]),
    "metrics_by_excess_actions": run["metrics_by_excess_actions"],
}, indent=2) + "\n")
(r / "training_complete.json").write_text(json.dumps({
    "status": "completed",
    "row": sys.argv[2],
    "recipe": "compiled_domain_lm_baseline",
    "domain": sys.argv[4],
    "dataset": sys.argv[4],
    "d_model": int(sys.argv[6]),
    "seed": int(sys.argv[7]),
    "learning_rate": float(sys.argv[8]),
}, indent=2) + "\n")
PY
