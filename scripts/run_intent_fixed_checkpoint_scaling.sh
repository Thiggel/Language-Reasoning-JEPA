#!/usr/bin/env bash
# Fixed-checkpoint test-time-scaling evaluation for intent-phrase JEPA.
set -euo pipefail

[[ -n "${RUN_DIR:-}" && -n "${TEXTJEPA_ROOT:-}" ]] || {
  echo "RUN_DIR and TEXTJEPA_ROOT are required" >&2
  exit 2
}
python_bin=${1:?python executable}
checkpoint=${2:?checkpoint path}
device=${DEVICE:-cuda:0}
episodes=${N_EPISODES:-200}
seed=${SEED:-321}
# depth:max_expand. Depth > 1 is deliberately candidate-privileged because
# the reference graph supplies future feasible actions.
cells=${SCALING_CELLS:-"1:64 2:64 4:64 8:64 2:4 2:8 4:4 4:8"}
controls=${SCALING_CONTROLS:-"model"}
slacks=${SCALING_SLACKS:-"0 2"}

[[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }
sha256sum "$checkpoint" > "$RUN_DIR/checkpoint.sha256"

for control in $controls; do
  simulator=latent
  score_control=$control
  energy=value
  case "$control" in
    model|zero|shuffle) ;;
    symbolic) simulator=symbolic; score_control=model ;;
    oracle_goal) score_control=model; energy=oracle_goal ;;
    symbolic_oracle_goal)
      simulator=symbolic; score_control=model; energy=oracle_goal ;;
    symbolic_exact_distance)
      simulator=symbolic; score_control=model; energy=symbolic_distance ;;
    *) echo "invalid scaling control: $control" >&2; exit 2;;
  esac
  for cell in $cells; do
    depth=${cell%%:*}
    cap=${cell##*:}
    case "$depth:$cap:$episodes:$seed" in
      *[!0-9:]*) echo "invalid scaling cell or numeric setting: $cell" >&2; exit 2;;
    esac
    oracle=false
    if (( depth > 1 )); then oracle=true; fi
    for slack in $slacks; do
      case "$slack" in *[!0-9]*) echo "invalid slack: $slack" >&2; exit 2;; esac
      stem="$RUN_DIR/control_${control}_depth${depth}_cap${cap}_slack${slack}"
      "$python_bin" "$TEXTJEPA_ROOT/scripts/plan.py" \
        "ckpt=$checkpoint" "device=$device" split=val \
        "n_episodes=$episodes" "seed=$seed" "slack=$slack" \
        "lookahead=$depth" "max_expand=$cap" \
        "allow_oracle_future_actions=$oracle" energy=value \
        "score_control=$score_control" "simulator=$simulator" "energy=$energy" \
        measure_flops=true "out=${stem}.json" "compute_out=${stem}_compute.json"
    done
  done
done

"$python_bin" - "$RUN_DIR" "$checkpoint" "$cells" "$controls" "$slacks" "$episodes" "$seed" <<'PY'
import hashlib, json, pathlib, sys

run = pathlib.Path(sys.argv[1])
checkpoint = pathlib.Path(sys.argv[2])
cells = sys.argv[3].split()
controls = sys.argv[4].split()
slacks = tuple(map(int, sys.argv[5].split()))
summary = {
    "protocol": "balanced-fixed-depth-absorbing-v2",
    "candidate_privilege": {
        "depth_1": "common currently feasible action menu",
        "depth_gt_1": "symbolic-future-action-tree: seeded reference-graph future feasible rollouts",
    },
    "validity_controls": controls,
    "invariants": {
        "first_action_budget": "equal up to one candidate",
        "terminal_handling": "absorbing no-op to fixed requested depth",
        "candidate_order": "seeded random permutation at every decision",
        "path_length_offset": "constant requested horizon across candidates",
    },
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "n_episodes": int(sys.argv[6]),
    "seed": int(sys.argv[7]),
    "controls": {},
}
for control in controls:
    summary["controls"][control] = {}
    for cell in cells:
        depth, cap = map(int, cell.split(":"))
        key = f"depth{depth}_cap{cap}"
        summary["controls"][control][key] = {}
        for slack in slacks:
            stem = run / f"control_{control}_{key}_slack{slack}"
            metrics = json.loads(stem.with_suffix(".json").read_text())
            compute = json.loads(
                (run / f"control_{control}_{key}_slack{slack}_compute.json").read_text()
            )
            if compute["checkpoint_sha256"] != summary["checkpoint_sha256"]:
                raise RuntimeError(
                    f"checkpoint changed during evaluation at {control}/{key}"
                )
            summary["controls"][control][key][str(slack)] = {
                "metrics": metrics,
                "compute": compute,
            }
(run / "scaling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
