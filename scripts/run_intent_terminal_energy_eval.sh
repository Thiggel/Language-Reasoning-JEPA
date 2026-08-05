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
case "$composition" in cumulative|terminal|root) ;; *)
  echo "invalid transition Energy composition: $composition" >&2; exit 2;;
esac
for depth in 1 4 8 16; do
  oracle=false; [[ "$depth" -gt 1 ]] && oracle=true
  for slack in 0 2; do
    "$py" "$TEXTJEPA_ROOT/scripts/plan.py" ckpt="$checkpoint" \
      device="$device" split=val n_episodes="$episodes" slack="$slack" \
      lookahead="$depth" max_expand="$width" search_algorithm=beam \
      transition_energy_composition="$composition" \
      allow_oracle_future_actions="$oracle" \
      out="$RUN_DIR/terminal_depth${depth}_slack${slack}.json"
  done
done
"$py" - "$RUN_DIR" "$label" "$checkpoint" "$episodes" "$width" \
  "$composition" <<'PY'
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
    'search_algorithm':'global_beam',
    'transition_energy_composition':sys.argv[6],
    'n_episodes':int(sys.argv[4]), 'beam_width':int(sys.argv[5]),
    'metrics_by_depth_and_slack':curves,
},indent=2)+'\n')
PY
