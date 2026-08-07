#!/usr/bin/env bash
# Checkpoint-only evaluation of terminal Energy under true beam search.
# State-Energy checkpoints score the final predicted state. Transition-Energy
# checkpoints score only the final predicted transition. No training occurs.
# One generous-budget run per depth: the policy never reads the budget, so a
# slack=$MAX_SLACK run yields the exact success rate at every smaller slack
# (success_by_slack) plus per-episode excess steps.
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
invalid_action_mode=${INVALID_ACTION_MODE:-noop}
depths=${EVAL_DEPTHS:-"1 4 8 16"}
max_slack=${MAX_SLACK:-4}
prior_top_k=${PRIOR_TOP_K:-8}; prior_top_p=${PRIOR_TOP_P:-1.0}
prior_feasibility_gate=${PRIOR_FEASIBILITY_GATE:-false}
case "$composition" in cumulative|terminal|root) ;; *)
  echo "invalid transition Energy composition: $composition" >&2; exit 2;;
esac
for depth in $depths; do
  oracle=false
  if [[ "$depth" -gt 1 && "$candidate_interface" == feasible_menu ]]; then
    oracle=true
  fi
  "$py" "$TEXTJEPA_ROOT/scripts/plan.py" ckpt="$checkpoint" \
    device="$device" split=val n_episodes="$episodes" slack="$max_slack" \
    slack_curve=true \
    lookahead="$depth" max_expand="$width" search_algorithm="$search_algorithm" \
    transition_energy_composition="$composition" \
    hybrid_local_pruning="$hybrid_local_pruning" \
    candidate_interface="$candidate_interface" \
    invalid_action_mode="$invalid_action_mode" \
    prior_top_k="$prior_top_k" prior_top_p="$prior_top_p" \
    prior_feasibility_gate="$prior_feasibility_gate" \
    allow_oracle_future_actions="$oracle" \
    out="$RUN_DIR/terminal_depth${depth}_slackcurve.json"
done
"$py" - "$RUN_DIR" "$label" "$checkpoint" "$episodes" "$width" \
  "$composition" "$search_algorithm" "$hybrid_local_pruning" \
  "$candidate_interface" "$invalid_action_mode" "$depths" \
  "$max_slack" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); max_slack=int(sys.argv[12])
curves={}; slack_curves={}
for depth in map(int, sys.argv[11].split()):
    p=r/f'terminal_depth{depth}_slackcurve.json'
    run=next(iter(json.loads(p.read_text()).values()))
    slack_curves[str(depth)]={
        'success_by_slack':run['success_by_slack'],
        'excess_steps':run['excess_steps'],
    }
    scalars={k:v for k,v in run.items()
             if k not in ('success_by_slack','excess_steps')}
    # Legacy per-slack rows (success overridden from the exact curve) keep
    # existing collectors working; the full curve lives in slack_curves.
    curves[str(depth)]={
        str(s): {**scalars, 'slack': s,
                 'success': run['success_by_slack'][str(s)]}
        for s in range(max_slack+1)
    }
ckpt=pathlib.Path(sys.argv[3])
(r/'metrics.json').write_text(json.dumps({
    'label':sys.argv[2], 'checkpoint':str(ckpt),
    'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    'search_algorithm':sys.argv[7],
    'hybrid_local_pruning':sys.argv[8].lower() == 'true',
    'candidate_interface':sys.argv[9],
    'invalid_action_mode':sys.argv[10],
    'transition_energy_composition':sys.argv[6],
    'n_episodes':int(sys.argv[4]), 'beam_width':int(sys.argv[5]),
    'max_slack':max_slack,
    'metrics_by_depth_and_slack':curves,
    'slack_curves':slack_curves,
},indent=2)+'\n')
PY
