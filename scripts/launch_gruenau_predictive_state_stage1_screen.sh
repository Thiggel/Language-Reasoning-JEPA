#!/usr/bin/env bash
set -euo pipefail

# Stage 1 upper-half LoRA screen. Five token-matched cells read one shared
# token-block file and differ only in the auxiliary objective, so the NTP-only
# arm isolates what the predictive loss adds over identical LoRA capacity.

root=/vol/home-vol2/ml/laitenbf/TextJEPA
round=2026-08-12-qwen-stage1-lora-screen-v1
token_blocks="$root/runs/autonomy/predictive_state/$round/_data/wikitext103_qwen_ctx1024.pt"

job_ids=(
  qwen05-lora-full-s0-v1
  qwen05-lora-ntp-only-s0-v1
  qwen05-lora-no-action-s0-v1
  qwen05-lora-action-only-s0-v1
  qwen05-lora-full-scale0.1-s0-v1
)
variants=(full ntp_only no_action action_only full)
scale_weights=(0.01 0.01 0.01 0.01 0.1)
hosts=(11 11 11 11 9)
gpus=(0 1 2 3 2)

cd "$root"

count=${#job_ids[@]}
for array in variants scale_weights hosts gpus; do
  eval "length=\${#${array}[@]}"
  [[ "$length" -eq "$count" ]] || {
    echo "cell arrays differ in length ($array)" >&2
    exit 2
  }
done
[[ -s "$token_blocks" ]] || {
  echo "shared token-block file is missing: $token_blocks" >&2
  exit 2
}

# The predictive-state sources must be pristine; unrelated subprojects may hold
# working-tree edits, and `git archive` reads the commit rather than the tree.
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
mkdir -p "$inventory_dir"
# Preserve the authoritative cluster-wide view immediately before placement.
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
    $1 == selected {
      found=1
      print (($2 + 0 < 1024 && $3 + 0 < 10) ? "FREE" : "BUSY")
    }
    END { if (!found) print "MISSING" }
  ' "$inventory")
  if [[ "$admission" != FREE ]]; then
    echo "refusing busy GPU gruenau${selected_host}:${selected_gpu} admission=$admission" >&2
    return 3
  fi
}

for index in "${!job_ids[@]}"; do
  job_id=${job_ids[$index]}
  variant=${variants[$index]}
  scale_weight=${scale_weights[$index]}
  host=${hosts[$index]}
  gpu=${gpus[$index]}
  # Repeat both low-memory and low-utilization admission immediately before
  # each launch; this narrows but cannot remove the shared-cluster race.
  check_gpu "$host" "$gpu"
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
export PREDICTIVE_STATE_SCALE_WEIGHT=$(printf %q "$scale_weight")
export PREDICTIVE_STATE_STEPS=1220
export PREDICTIVE_STATE_MICROBATCH=8
export PREDICTIVE_STATE_ACCUMULATION=2
export PREDICTIVE_STATE_SEQUENCE_LENGTH=1024
export PREDICTIVE_STATE_EVAL_EVERY=100
export PREDICTIVE_STATE_EVAL_BATCHES=16
export PREDICTIVE_STATE_DTYPE=bfloat16
mkdir -p "\$TMPDIR"
cd "\$snapshot"
"\$TEXTJEPA_PYTHON" - "\$run_dir" "\$run_id" <<'PY'
import json,os,pathlib,platform,socket,sys,torch
path=pathlib.Path(sys.argv[1])
(path/'resolved_config.json').write_text(json.dumps({
  'run_id':sys.argv[2], 'source_revision':'$revision',
  'model_id':'Qwen/Qwen2.5-0.5B',
  'model_revision':'060db6499f32faf8b98477b0a26969ef7d8b9987',
  'variant':'$variant', 'backbone_mode':'lora',
  'target_layer':12, 'source_layers':[18,24], 'lora_first_layer':13,
  'lora_rank':16, 'lora_alpha':32,
  'steps':1220, 'microbatch_size':8, 'gradient_accumulation':2,
  'context_length':1024, 'target_tokens':20000000,
  'prediction_weight':0.1, 'scale_weight':float('$scale_weight'),
  'predictor_learning_rate':3e-4, 'lora_learning_rate':1e-4,
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
timeout --signal=TERM --kill-after=120 21600 \
  bash scripts/run_predictive_state_stage1_screen.sh $(printf %q "$variant") 0 \
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
    'schema_version': 1,
    'run_id': sys.argv[2],
    'status': sys.argv[3],
    'exit_code': int(sys.argv[4]),
    'scientific_validity': 'not_admitted',
    'failure_summary': 'see stderr.log and stdout.log',
}, indent=2) + '\n')
PY
fi
exit "\$code"
EOF
  chmod +x "$job"
  printf -v remote '%q ' env CUDA_VISIBLE_DEVICES="$gpu" bash "$job"
  launcher_log="$run_dir/launcher.log"
  command="nohup $remote > $(printf %q "$launcher_log") 2>&1 < /dev/null & echo \$!"
  pid=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
    "laitenbf@gruenau${host}.informatik.hu-berlin.de" "$command")
  printf '%s\thost=gruenau%s\tgpu=%s\tpid=%s\trevision=%s\n' \
    "$job_id" "$host" "$gpu" "$pid" "$revision"
done
