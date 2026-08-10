#!/usr/bin/env bash
set -euo pipefail

root=/vol/home-vol2/ml/laitenbf/TextJEPA
round=2026-08-10-qwen-frozen-transition-diagnostic-v1
host=1
variants=(full no_action)
gpus=(0 1)
cd "$root"

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

# The caller must run gruenau-gpus immediately before this script. Repeat a
# direct device check to narrow, but not eliminate, the placement race.
ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
  "laitenbf@gruenau${host}.informatik.hu-berlin.de" \
  "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits" \
  > "/tmp/predictive-state-gruenau${host}-inventory.txt"

for index in "${!variants[@]}"; do
  variant=${variants[$index]}
  gpu=${gpus[$index]}
  job_id="qwen05-frozen-${variant//_/-}-s0-v1"
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
