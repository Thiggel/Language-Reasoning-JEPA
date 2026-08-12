#!/usr/bin/env bash
set -euo pipefail

# Stage 1 pressure round. The screen found that the auxiliary objective changes
# the representation about 250x less than the next-token loss it rides along
# with, while the predictor solves the transition on its own parameters. Two
# ladders test the two ways to raise the pressure. Both are anchored on the
# previous round's `qwen05-lora-full-scale0.1-s0-v1` cell, which is exactly
# their shared corner: prediction weight 0.1, scale weight 0.1, and the full
# 448-wide projection. Scale weight is 0.1 throughout because the screen showed
# it calibrates activation scale for free (RMS ratio 1.0006 against 1.009).
#
#   weight ladder:     0.3, 1.0, 3.0   at the protocol projection
#   bottleneck ladder: 32, 8           at the protocol prediction weight
#   linear predictor:  no SwiGLU, full-rank linear map only
#   single-state:      condition on layer 18 or layer 24 alone
#   plus projection 8 with the action channel held at full 448 width, which
#   separates state scarcity from action legibility
#
# Everything else matches the screen exactly: same shared token blocks, same
# batch order, same 20M tokens, same seed.

root=/vol/home-vol2/ml/laitenbf/TextJEPA
round=2026-08-12-qwen-stage1-pressure-v1
screen_round=2026-08-12-qwen-stage1-lora-screen-v1
token_blocks="$root/runs/autonomy/predictive_state/$screen_round/_data/wikitext103_qwen_ctx1024.pt"

job_ids=(
  qwen05-lora-full-lpred0.3-s0-v1
  qwen05-lora-full-lpred1.0-s0-v1
  qwen05-lora-full-lpred3.0-s0-v1
  qwen05-lora-full-proj32-s0-v1
  qwen05-lora-full-proj8-s0-v1
  qwen05-lora-full-proj8-action448-s0-v1
  qwen05-lora-full-linear-s0-v1
  qwen05-lora-full-src18only-s0-v1
  qwen05-lora-full-src24only-s0-v1
)
prediction_weights=(0.3 1.0 3.0 0.1 0.1 0.1 0.1 0.1 0.1)
# 896 keeps the linear map full rank; 672 makes a single-source predictor
# exactly parameter matched to the two-source one (8,830,976) at the same
# 1344-wide concatenation, so dropping a state is the only change.
projection_sizes=("" "" "" 32 8 8 896 672 672)
# The shared projection width throttles the realized action along with the
# state. The last cell repeats the tightest bottleneck with the action
# channel held at full width, isolating state scarcity from action legibility.
action_projection_sizes=("" "" "" "" "" 448 "" "" "")
linear_predictors=("" "" "" "" "" "" 1 "" "")
source_layer_sets=("" "" "" "" "" "" "" 18 24)
# Cells sharing a host:gpu run sequentially in one chained job, so a round is
# not limited to the number of simultaneously free devices. Each cell still
# gets its own run directory, state file and exit marker, and a failed cell
# does not stop the ones queued behind it.
hosts=(11 11 11 9 11 11 11 11 9)
gpus=(0 2 0 2 2 0 2 0 2)

cd "$root"

count=${#job_ids[@]}
for array in prediction_weights projection_sizes action_projection_sizes \
             linear_predictors source_layer_sets hosts gpus; do
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

placements=()
for index in "${!job_ids[@]}"; do
  placements+=("${hosts[$index]}:${gpus[$index]}")
done
devices=$(printf '%s\n' "${placements[@]}" | sort -u)

for device in $devices; do
  host=${device%%:*}
  gpu=${device##*:}
  chain_dir="$root/runs/autonomy/predictive_state/$round/_chain"
  mkdir -p "$chain_dir"
  chain="$chain_dir/gruenau${host}-gpu${gpu}.sh"
  printf '#!/usr/bin/env bash\nset -uo pipefail\n' > "$chain"
  chain_members=()

for index in "${!job_ids[@]}"; do
  [[ "${hosts[$index]}:${gpus[$index]}" == "$device" ]] || continue
  job_id=${job_ids[$index]}
  prediction_weight=${prediction_weights[$index]}
  projection_size=${projection_sizes[$index]}
  action_projection_size=${action_projection_sizes[$index]}
  linear_predictor=${linear_predictors[$index]}
  source_layer_set=${source_layer_sets[$index]}
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
export PREDICTIVE_STATE_PREDICTION_WEIGHT=$(printf %q "$prediction_weight")
export PREDICTIVE_STATE_SCALE_WEIGHT=0.1
export PREDICTIVE_STATE_STEPS=1220
export PREDICTIVE_STATE_MICROBATCH=8
export PREDICTIVE_STATE_ACCUMULATION=2
export PREDICTIVE_STATE_SEQUENCE_LENGTH=1024
export PREDICTIVE_STATE_EVAL_EVERY=100
export PREDICTIVE_STATE_EVAL_BATCHES=16
export PREDICTIVE_STATE_DTYPE=bfloat16
EOF
  if [[ -n "$projection_size" ]]; then
    printf 'export PREDICTIVE_STATE_PROJECTION_SIZE=%q\n' "$projection_size" >> "$job"
  fi
  if [[ -n "$action_projection_size" ]]; then
    printf 'export PREDICTIVE_STATE_ACTION_PROJECTION_SIZE=%q\n' \
      "$action_projection_size" >> "$job"
  fi
  if [[ -n "$linear_predictor" ]]; then
    printf 'export PREDICTIVE_STATE_LINEAR_PREDICTOR=1\n' >> "$job"
  fi
  if [[ -n "$source_layer_set" ]]; then
    printf 'export PREDICTIVE_STATE_SOURCE_LAYERS=%q\n' "$source_layer_set" >> "$job"
  fi
  cat >> "$job" <<EOF
mkdir -p "\$TMPDIR"
cd "\$snapshot"
"\$TEXTJEPA_PYTHON" - "\$run_dir" "\$run_id" <<'PY'
import json,os,pathlib,platform,socket,sys,torch
path=pathlib.Path(sys.argv[1])
projection=os.environ.get('PREDICTIVE_STATE_PROJECTION_SIZE')
(path/'resolved_config.json').write_text(json.dumps({
  'run_id':sys.argv[2], 'source_revision':'$revision',
  'model_id':'Qwen/Qwen2.5-0.5B',
  'model_revision':'060db6499f32faf8b98477b0a26969ef7d8b9987',
  'variant':'full', 'backbone_mode':'lora',
  'target_layer':12, 'source_layers':[18,24], 'lora_first_layer':13,
  'lora_rank':16, 'lora_alpha':32,
  'steps':1220, 'microbatch_size':8, 'gradient_accumulation':2,
  'context_length':1024, 'target_tokens':20000000,
  'prediction_weight':float('$prediction_weight'), 'scale_weight':0.1,
  'projection_size':None if projection is None else int(projection),
  'linear_predictor':os.environ.get('PREDICTIVE_STATE_LINEAR_PREDICTOR') is not None,
  'source_layers_override':os.environ.get('PREDICTIVE_STATE_SOURCE_LAYERS'),
  'action_projection_size':(
    None if os.environ.get('PREDICTIVE_STATE_ACTION_PROJECTION_SIZE') is None
    else int(os.environ['PREDICTIVE_STATE_ACTION_PROJECTION_SIZE'])),
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
  # Admission is checked once per device, immediately before the chain starts.
  # Later cells in a chain inherit the device from the cell ahead of them, so
  # they cannot race a third party for it. A busy device defers its own chain
  # only; it must not abort placement onto the remaining devices.
  if ! check_gpu "$host" "$gpu"; then
    printf 'deferred\thost=gruenau%s\tgpu=%s\tcells=%s\n' \
      "$host" "$gpu" "$(IFS=,; echo "${chain_members[*]}")"
    continue
  fi
  printf -v remote '%q ' env CUDA_VISIBLE_DEVICES="$gpu" bash "$chain"
  launcher_log="$chain_dir/gruenau${host}-gpu${gpu}.log"
  command="nohup $remote > $(printf %q "$launcher_log") 2>&1 < /dev/null & echo \$!"
  pid=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
    "laitenbf@gruenau${host}.informatik.hu-berlin.de" "$command")
  printf 'host=gruenau%s\tgpu=%s\tpid=%s\tcells=%s\trevision=%s\n' \
    "$host" "$gpu" "$pid" "$(IFS=,; echo "${chain_members[*]}")" "$revision"
done
