#!/usr/bin/env bash
set -euo pipefail

# Stage 1 objective-balance round.
#
# The pressure round raised the prediction weight thirtyfold and concluded that
# the objective cannot shape the backbone. That conclusion was overstated: the
# next-token term carries a hard-coded weight of one, so even at prediction
# weight 3.0 the measured loss was still 8.2x next-token dominated. The
# auxiliary objective was never the primary thing being optimized.
#
# This round tests the three untested directions:
#   - downweight or remove next-token prediction, so the predictive loss leads;
#   - adapt the upper stack fully instead of through a rank-16 adapter;
#   - train four times as long under the most promising of those.
#
# Layers 1-12 and the embeddings stay frozen in every cell, including the full
# finetuning ones, so the layer-12 target remains a fixed anchor. A collapsing
# source representation therefore *raises* the transition loss and cannot be a
# trivial solution.
#
# Read cosine as the outcome and NLL as its price. If cosine falls well below
# the 0.1050 the round-two cells reached, the representation can be shaped and
# Stage 1 was mis-framed as a free improvement rather than a trade-off. If it
# does not move even with next-token prediction switched off entirely, the
# residual error is irreducible conditional variance and no weighting, capacity
# or duration will change it.

root=/vol/home-vol2/ml/laitenbf/TextJEPA
round=2026-08-13-qwen-stage1-objective-balance-v1
screen_round=2026-08-12-qwen-stage1-lora-screen-v1
token_blocks="$root/runs/autonomy/predictive_state/$screen_round/_data/wikitext103_qwen_ctx1024.pt"

job_ids=(
  qwen05-lora-ntp0.01-lpred1-s0-v1
  qwen05-lora-ntp0-lpred1-s0-v1
  qwen05-fullft-ntp1-lpred0.1-s0-v1
  qwen05-fullft-ntp0.01-lpred1-s0-v1
  qwen05-fullft-ntp0.01-lpred1-lr1e-4-s0-v1
  qwen05-fullft-ntp0.01-lpred1-4x-s0-v1
)
ntp_weights=(0.01 0.0 1.0 0.01 0.01 0.01)
prediction_weights=(1.0 1.0 0.1 1.0 1.0 1.0)
backbone_modes=(lora lora full_upper full_upper full_upper full_upper)
backbone_rates=("" "" 2e-5 2e-5 1e-4 2e-5)
step_counts=(1220 1220 1220 1220 1220 4880)
hosts=(11 11 11 9 11 11)
gpus=(0 2 0 2 2 0)

cd "$root"

count=${#job_ids[@]}
for array in ntp_weights prediction_weights backbone_modes backbone_rates \
             step_counts hosts gpus; do
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

placements=()
for index in "${!job_ids[@]}"; do
  placements+=("${hosts[$index]}:${gpus[$index]}")
done
devices=$(printf '%s\n' "${placements[@]}" | sort -u)

deferred_dir="$inventory_dir/_chain"
mkdir -p "$deferred_dir"

for device in $devices; do
  host=${device%%:*}
  gpu=${device##*:}
  chain="$deferred_dir/gruenau${host}-gpu${gpu}.sh"
  printf '#!/usr/bin/env bash\nset -uo pipefail\n' > "$chain"
  chain_members=()
  chain_jobs=()

for index in "${!job_ids[@]}"; do
  [[ "${hosts[$index]}:${gpus[$index]}" == "$device" ]] || continue
  job_id=${job_ids[$index]}
  ntp_weight=${ntp_weights[$index]}
  prediction_weight=${prediction_weights[$index]}
  backbone_mode=${backbone_modes[$index]}
  backbone_rate=${backbone_rates[$index]}
  steps=${step_counts[$index]}
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
  chain_members+=("$job_id")
  chain_jobs+=("$job")
  printf 'bash %q\n' "$job" >> "$chain"
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
export PREDICTIVE_STATE_NTP_WEIGHT=$(printf %q "$ntp_weight")
export PREDICTIVE_STATE_PREDICTION_WEIGHT=$(printf %q "$prediction_weight")
export PREDICTIVE_STATE_BACKBONE_MODE=$(printf %q "$backbone_mode")
export PREDICTIVE_STATE_SCALE_WEIGHT=0.1
export PREDICTIVE_STATE_STEPS=$(printf %q "$steps")
export PREDICTIVE_STATE_MICROBATCH=8
export PREDICTIVE_STATE_ACCUMULATION=2
export PREDICTIVE_STATE_SEQUENCE_LENGTH=1024
export PREDICTIVE_STATE_EVAL_EVERY=100
export PREDICTIVE_STATE_EVAL_BATCHES=16
export PREDICTIVE_STATE_DTYPE=bfloat16
EOF
  if [[ -n "$backbone_rate" ]]; then
    printf 'export PREDICTIVE_STATE_BACKBONE_LR=%q\n' "$backbone_rate" >> "$job"
  fi
  cat >> "$job" <<EOF
mkdir -p "\$TMPDIR"
cd "\$snapshot"
"\$TEXTJEPA_PYTHON" - "\$run_dir" "\$run_id" <<'PY'
import json,os,pathlib,platform,socket,sys,torch
path=pathlib.Path(sys.argv[1])
rate=os.environ.get('PREDICTIVE_STATE_BACKBONE_LR')
(path/'resolved_config.json').write_text(json.dumps({
  'run_id':sys.argv[2], 'source_revision':'$revision',
  'model_id':'Qwen/Qwen2.5-0.5B',
  'model_revision':'060db6499f32faf8b98477b0a26969ef7d8b9987',
  'variant':'full', 'backbone_mode':'$backbone_mode',
  'target_layer':12, 'source_layers':[18,24], 'lora_first_layer':13,
  'steps':int('$steps'), 'microbatch_size':8, 'gradient_accumulation':2,
  'context_length':1024, 'target_tokens':int('$steps')*16384,
  'ntp_weight':float('$ntp_weight'),
  'prediction_weight':float('$prediction_weight'), 'scale_weight':0.1,
  'predictor_learning_rate':3e-4,
  'backbone_learning_rate':None if rate is None else float(rate),
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
timeout --signal=TERM --kill-after=120 43200 \
  bash scripts/run_predictive_state_stage1_screen.sh full 0 \
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
exit 0
EOF
  chmod +x "$job"
done

  [[ ${#chain_members[@]} -gt 0 ]] || continue
  chmod +x "$chain"
  # A busy device defers its own chain to the waiter rather than aborting the
  # round; the waiter re-applies this same gate before each cell starts.
  if ! check_gpu "$host" "$gpu"; then
    waiter_log="$deferred_dir/deferred-gruenau${host}-gpu${gpu}.log"
    printf -v deferred '%q ' bash "$root/scripts/wait_and_run_predictive_state_cells.sh" \
      "$gpu" "${chain_jobs[@]}"
    pid=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
      "laitenbf@gruenau${host}.informatik.hu-berlin.de" \
      "nohup $deferred > $(printf %q "$waiter_log") 2>&1 < /dev/null & echo \$!")
    printf 'deferred\thost=gruenau%s\tgpu=%s\tpid=%s\tcells=%s\n' \
      "$host" "$gpu" "$pid" "$(IFS=,; echo "${chain_members[*]}")"
    continue
  fi
  printf -v remote '%q ' env CUDA_VISIBLE_DEVICES="$gpu" bash "$chain"
  launcher_log="$deferred_dir/gruenau${host}-gpu${gpu}.log"
  command="nohup $remote > $(printf %q "$launcher_log") 2>&1 < /dev/null & echo \$!"
  pid=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
    "laitenbf@gruenau${host}.informatik.hu-berlin.de" "$command")
  printf 'host=gruenau%s\tgpu=%s\tpid=%s\tcells=%s\trevision=%s\n' \
    "$host" "$gpu" "$pid" "$(IFS=,; echo "${chain_members[*]}")" "$revision"
done
