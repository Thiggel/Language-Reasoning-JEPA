#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?existing checkpoint}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

"$python_bin" "$TEXTJEPA_ROOT/scripts/eval_multiscale_edit_mpc.py" \
  --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
  --regime id --examples 1 --horizon 4 --max-actions 4 \
  --beam-width 4 --top-positions 4 --top-tokens 4 --max-candidates 16 \
  --out "$RUN_DIR/primitive_macro_rerank_smoke.json"

for mode in subgoal decoder_open_loop decoder decoder_refine; do
  "$python_bin" "$TEXTJEPA_ROOT/scripts/eval_hierarchical_edit_cem.py" \
    --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
    --mode "$mode" --regime id --examples 1 --max-actions 4 \
    --high-horizon 1 --low-horizon 4 --cem-candidates 8 \
    --cem-iterations 2 --cem-elites 2 --reachability-topk 2 \
    --beam-width 4 --top-positions 4 --top-tokens 4 \
    --out "$RUN_DIR/${mode}_smoke.json"
done
