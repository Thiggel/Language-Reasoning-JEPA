#!/usr/bin/env bash
# Checkpoint-only evaluation of terminal Energy under true beam search.
# State-Energy checkpoints score the final predicted state. Transition-Energy
# checkpoints score only the final predicted transition. No training occurs.
set -euo pipefail
[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2; exit 2;
}
py=${1:?python executable}; checkpoint=${2:?checkpoint}; label=${3:?label}
device=${DEVICE:-cuda:0}; episodes=${N_EPISODES:-200}; width=${BEAM_WIDTH:-8}
composition=${TRANSITION_ENERGY_COMPOSITION:-terminal}
search_algorithm=${SEARCH_ALGORITHM:-beam}
hybrid_local_pruning=${HYBRID_LOCAL_PRUNING:-false}
candidate_interface=${CANDIDATE_INTERFACE:-feasible_menu}
case "$composition" in cumulative|terminal|root) ;; *)
  echo "invalid transition Energy composition: $composition" >&2; exit 2;;
esac
for depth in 1 4 8 16; do
  oracle=false
  if [[ "$depth" -gt 1 && "$candidate_interface" == feasible_menu ]]; then
    oracle=true
  fi
  for slack in 0 2; do
    "$py" "$TEXTJEPA_ROOT/scripts/plan.py" ckpt="$checkpoint" \
      device="$device" split=val n_episodes="$episodes" slack="$slack" \
      lookahead="$depth" max_expand="$width" search_algorithm="$search_algorithm" \
      transition_energy_composition="$composition" \
      hybrid_local_pruning="$hybrid_local_pruning" \
      candidate_interface="$candidate_interface" \
      allow_oracle_future_actions="$oracle" \
      out="$RUN_DIR/terminal_depth${depth}_slack${slack}.json"
  done
done
"$py" - "$RUN_DIR" "$label" "$checkpoint" "$episodes" "$width" \
  "$composition" "$search_algorithm" "$hybrid_local_pruning" \
  "$candidate_interface" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); curves={}
for depth in (1,4,8,16):
    curves[str(depth)]={}
    for slack in (0,2):
        p=r/f'terminal_depth{depth}_slack{slack}.json'
        curves[str(depth)][str(slack)]=next(iter(json.loads(p.read_text()).values()))
ckpt=pathlib.Path(sys.argv[3])
(r/'metrics.json').write_text(json.dumps({
    'label':sys.argv[2], 'checkpoint':str(ckpt),
    'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    'search_algorithm':sys.argv[7],
    'hybrid_local_pruning':sys.argv[8].lower() == 'true',
    'candidate_interface':sys.argv[9],
    'transition_energy_composition':sys.argv[6],
    'n_episodes':int(sys.argv[4]), 'beam_width':int(sys.argv[5]),
    'metrics_by_depth_and_slack':curves,
},indent=2)+'\n')
PY
