#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python executable}
checkpoint=${2:?existing checkpoint}
mode=${3:?no_prior, detached_prior, or attached_prior}
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR must be supplied by researchctl" >&2
  exit 2
fi

eval_args=()
if [[ "$mode" == no_prior ]]; then
  eval_args+=(--disable-base-prior --macro-prior-weight 0)
fi

"$python_bin" "$TEXTJEPA_ROOT/scripts/eval_multiscale_edit_mpc.py" \
  --ckpt "$checkpoint" --device "${DEVICE:-cuda:0}" \
  --regime id --examples 1 --horizon 1 --beam-width 4 \
  --top-positions 4 --top-tokens 4 --max-candidates 16 \
  --out "$RUN_DIR/mpc_h1_smoke.json" "${eval_args[@]}"
