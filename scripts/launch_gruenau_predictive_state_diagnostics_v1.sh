#!/usr/bin/env bash
set -euo pipefail

root=/vol/home-vol2/ml/laitenbf/TextJEPA
round=2026-08-10-qwen-frozen-transition-diagnostic-v5
host=1
variants=(full action_only)
gpus=(0 1)
cd "$root"

[[ ${#variants[@]} -eq ${#gpus[@]} ]] || {
  echo "variant/GPU arrays differ in length" >&2
  exit 2
}

[[ -z "$(git status --porcelain)" ]] || {
  echo "refusing to snapshot a dirty worktree" >&2
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
  local selected_gpu=$1
  local inventory="$inventory_dir/gruenau${host}-gpu${selected_gpu}-direct.csv"
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
    "laitenbf@gruenau${host}.informatik.hu-berlin.de" \
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
    echo "refusing busy GPU gruenau${host}:${selected_gpu} admission=$admission" >&2
    return 3
  fi
}

for index in "${!variants[@]}"; do
  variant=${variants[$index]}
  gpu=${gpus[$index]}
  # Repeat both low-memory and low-utilization admission immediately before
  # each launch; this narrows but cannot remove the shared-cluster race.
  check_gpu "$gpu"
  job_id="qwen05-frozen-${variant//_/-}-capacity-matched-s0-v5"
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
mkdir -p "\$TMPDIR"
cd "\$snapshot"
"\$TEXTJEPA_PYTHON" - "\$run_dir" "\$run_id" <<'PY'
import json,pathlib,platform,socket,sys,torch
path=pathlib.Path(sys.argv[1])
(path/'resolved_config.json').write_text(json.dumps({
  'run_id':sys.argv[2], 'source_revision':'$revision',
  'model_id':'Qwen/Qwen2.5-0.5B',
  'model_revision':'060db6499f32faf8b98477b0a26969ef7d8b9987',
  'variant':'$variant', 'backbone_mode':'frozen', 'steps':200,
  'context_length':256, 'seed':0, 'dtype':'float16'
},indent=2)+'\n')
(path/'environment.json').write_text(json.dumps({
  'hostname':socket.gethostname(), 'platform':platform.platform(),
  'python':sys.version, 'torch':torch.__version__,
  'cuda_visible_devices':__import__('os').environ.get('CUDA_VISIBLE_DEVICES')
},indent=2)+'\n')
PY
set +e
timeout --signal=TERM --kill-after=120 14400 \
  bash scripts/run_predictive_state_frozen_diagnostic.sh $(printf %q "$variant") 0 \
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
