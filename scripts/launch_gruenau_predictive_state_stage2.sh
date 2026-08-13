#!/usr/bin/env bash
set -euo pipefail

# Stage 2: recurrent execution. Decode by predicting the layer-12 residual and
# running only layers 13-24, so the lower half never runs again after prefill.
#
# The untrained open-loop measurement on the Stage 1 checkpoints already shows
# the shape of the problem. State error does not diverge: 0.1436 at horizon 1
# against 0.1405 at horizon 256. Measured throughput is 1.86x and decode cache
# growth halves, both past their gates. What fails is fidelity: excess NLL
# saturates near 1.12 nats against a 0.3 gate, and top-1 agreement falls to
# about 0.49. The curriculum therefore has to remove a stable bias rather than
# arrest a blow-up, which is the better of the two problems to have.

root=/vol/home-vol2/ml/laitenbf/TextJEPA
round=2026-08-13-qwen-stage2-curriculum-v1
stage1_round=2026-08-13-qwen-stage1-fineweb-scale-v1
token_blocks="$root/runs/autonomy/predictive_state/_corpora/fineweb_edu_qwen_ctx1024.pt"

job_ids=(
  qwen05-stage2-from-lpred3-s0-v1
  qwen05-stage2-from-lpred1-ntp0.1-s0-v1
)
stage1_cells=(
  qwen05-fwe-full-lpred3-ntp1-s0-v1
  qwen05-fwe-full-lpred1-ntp0.1-s0-v1
)
hosts=(11 11)
gpus=(2 3)

cd "$root"

count=${#job_ids[@]}
for array in stage1_cells hosts gpus; do
  eval "length=\${#${array}[@]}"
  [[ "$length" -eq "$count" ]] || { echo "cell arrays differ ($array)" >&2; exit 2; }
done
[[ -s "$token_blocks" ]] || { echo "missing token blocks: $token_blocks" >&2; exit 2; }

dirty=$(git status --porcelain -- \
  src/textjepa/models/action_transition.py \
  src/textjepa/objectives/predictive_state.py \
  src/textjepa/data/predictive_state.py \
  src/textjepa/training/predictive_state.py \
  src/textjepa/analysis/predictive_state.py \
  src/textjepa/planning/predictive_state.py \
  'scripts/*predictive_state*' 'scripts/*action_transition*' \
  configs/predictive_state)
[[ -z "$dirty" ]] || {
  echo "refusing to snapshot: predictive-state sources are dirty" >&2
  printf '%s\n' "$dirty" >&2
  exit 2
}
revision=$(git rev-parse HEAD)
snapshot="$root/runs/autonomy/_code/$revision"
if [[ ! -d "$snapshot" ]]; then
  temporary=$(mktemp -d "$root/runs/autonomy/_code/.tmp-${revision}.XXXXXX")
  git archive "$revision" | tar -x -C "$temporary"
  mv "$temporary" "$snapshot"
fi

inventory_dir="$root/runs/autonomy/predictive_state/$round"
mkdir -p "$inventory_dir/_chain"
gruenau-gpus > "$inventory_dir/gruenau-gpus-before-launch.txt"

check_gpu() {
  local selected_host=$1 selected_gpu=$2
  local inventory="$inventory_dir/gruenau${selected_host}-gpu${selected_gpu}-direct.csv"
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
    "laitenbf@gruenau${selected_host}.informatik.hu-berlin.de" \
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits" \
    > "$inventory"
  local admission
  admission=$(awk -F, -v selected="$selected_gpu" '
    { for (i=1; i<=3; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
    $1 == selected { found=1; print (($2 + 0 < 1024 && $3 + 0 < 10) ? "FREE" : "BUSY") }
    END { if (!found) print "MISSING" }
  ' "$inventory")
  if [[ "$admission" != FREE ]]; then
    echo "refusing busy GPU gruenau${selected_host}:${selected_gpu} admission=$admission" >&2
    return 3
  fi
}

for index in "${!job_ids[@]}"; do
  job_id=${job_ids[$index]}
  stage1_cell=${stage1_cells[$index]}
  host=${hosts[$index]}
  gpu=${gpus[$index]}
  stage1_checkpoint="$root/runs/autonomy/predictive_state/$stage1_round/$stage1_cell/model/last.pt"
  [[ -s "$stage1_checkpoint" ]] || {
    echo "missing Stage 1 checkpoint: $stage1_checkpoint" >&2
    exit 2
  }
  run_dir="$root/runs/autonomy/predictive_state/$round/$job_id"
  if [[ -s "$run_dir/state" ]]; then
    state=$(tr -d '[:space:]' < "$run_dir/state")
    if [[ "$state" == RUNNING || "$state" == COMPLETED ]]; then
      echo "skip $job_id state=$state"
      continue
    fi
  fi
  mkdir -p "$run_dir"
  job="$run_dir/job.sh"
  cat > "$job" <<EOF
#!/usr/bin/env bash
set -uo pipefail
run_dir=$(printf %q "$run_dir")
run_id=$(printf %q "$job_id")
snapshot=$(printf %q "$snapshot")
mkdir -p "\$run_dir"
printf '%s\n' RUNNING > "\$run_dir/state"
printf '%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "\$run_dir/started_at"
export RUN_DIR="\$run_dir" RUN_ID="\$run_id" TEXTJEPA_ROOT="\$snapshot"
export TEXTJEPA_SOURCE_REVISION=$(printf %q "$revision")
export TEXTJEPA_PYTHON="$root/.venv/bin/python"
export PYTHONPATH="\$snapshot/src"
export HF_HOME=/vol/home-vol2/ml/laitenbf/.cache/textjepa/huggingface
export TMPDIR="\$run_dir/tmp-direct"
export PREDICTIVE_STATE_TOKEN_BLOCKS=$(printf %q "$token_blocks")
export PREDICTIVE_STATE_STAGE1_CHECKPOINT=$(printf %q "$stage1_checkpoint")
export PREDICTIVE_STATE_HORIZONS="1 4 8 16 32"
export PREDICTIVE_STATE_STAGE_STEPS="600 600 600 600 1000"
export PREDICTIVE_STATE_REPLAY_FRACTIONS="0.0 0.0 0.0 0.25 0.25"
export PREDICTIVE_STATE_MICROBATCH=4
export PREDICTIVE_STATE_ACCUMULATION=2
export PREDICTIVE_STATE_SEQUENCE_LENGTH=512
export PREDICTIVE_STATE_EVAL_BATCHES=8
export PREDICTIVE_STATE_DTYPE=bfloat16
mkdir -p "\$TMPDIR"
cd "\$snapshot"
"\$TEXTJEPA_PYTHON" - "\$run_dir" "\$run_id" <<'PY'
import json,os,pathlib,platform,socket,sys,torch
path=pathlib.Path(sys.argv[1])
(path/'resolved_config.json').write_text(json.dumps({
  'run_id':sys.argv[2], 'source_revision':'$revision', 'stage':'stage2',
  'model_id':'Qwen/Qwen2.5-0.5B',
  'stage1_checkpoint':os.environ['PREDICTIVE_STATE_STAGE1_CHECKPOINT'],
  'horizons':[1,4,8,16,32], 'stage_steps':[600,600,600,600,1000],
  'replay_fractions':[0.0,0.0,0.0,0.25,0.25],
  'microbatch_size':4, 'gradient_accumulation':2, 'sequence_length':512,
  'predictor_learning_rate':1e-4, 'backbone_learning_rate':5e-5,
  'seed':0, 'dtype':'bfloat16',
  'token_blocks':os.environ['PREDICTIVE_STATE_TOKEN_BLOCKS'],
},indent=2)+'\n')
(path/'environment.json').write_text(json.dumps({
  'hostname':socket.gethostname(), 'platform':platform.platform(),
  'python':sys.version, 'torch':torch.__version__,
  'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
  'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
},indent=2)+'\n')
PY
set +e
timeout --signal=TERM --kill-after=120 64800 \
  bash scripts/run_predictive_state_stage2_curriculum.sh \
  >"\$run_dir/stdout.log" 2>"\$run_dir/stderr.log"
code=\$?
set -e
printf '%s\n' "\$code" > "\$run_dir/exit_code"
if [[ "\$code" -eq 0 ]]; then state=COMPLETED
elif [[ "\$code" -eq 124 ]]; then state=TIMEOUT
else state=FAILED
fi
printf '%s\n' "\$state" > "\$run_dir/state"
if [[ ! -s "\$run_dir/run_summary.json" ]]; then
  "\$TEXTJEPA_PYTHON" - "\$run_dir" "\$run_id" "\$state" "\$code" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
(path / 'run_summary.json').write_text(json.dumps({
    'schema_version': 1, 'run_id': sys.argv[2], 'status': sys.argv[3],
    'exit_code': int(sys.argv[4]), 'scientific_validity': 'not_admitted',
    'failure_summary': 'see stderr.log and stdout.log',
}, indent=2) + '\n')
PY
fi
exit 0
EOF
  chmod +x "$job"
  if ! check_gpu "$host" "$gpu"; then
    waiter_log="$inventory_dir/_chain/deferred-gruenau${host}-gpu${gpu}.log"
    printf -v deferred '%q ' bash \
      "$root/scripts/wait_and_run_predictive_state_cells.sh" "$gpu" "$job"
    pid=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
      "laitenbf@gruenau${host}.informatik.hu-berlin.de" \
      "nohup $deferred > $(printf %q "$waiter_log") 2>&1 < /dev/null & echo \$!")
    printf 'deferred\thost=gruenau%s\tgpu=%s\tpid=%s\tcell=%s\n' \
      "$host" "$gpu" "$pid" "$job_id"
    continue
  fi
  printf -v remote '%q ' env CUDA_VISIBLE_DEVICES="$gpu" bash "$job"
  launcher_log="$run_dir/launcher.log"
  pid=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
    "laitenbf@gruenau${host}.informatik.hu-berlin.de" \
    "nohup $remote > $(printf %q "$launcher_log") 2>&1 < /dev/null & echo \$!")
  printf '%s\thost=gruenau%s\tgpu=%s\tpid=%s\trevision=%s\n' \
    "$job_id" "$host" "$gpu" "$pid" "$revision"
done
