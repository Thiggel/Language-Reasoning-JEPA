#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 {ema|sg|grad} {none|vic|sig} {off|on} [device]" >&2
  exit 2
fi

target=$1
regularizer=$2
ldad=$3
device=${4:-cuda:0}

case "$target" in
  ema) state_target=ema ;;
  sg) state_target=online ;;
  grad) state_target=online_nosg ;;
  *) echo "unknown target mode: $target" >&2; exit 2 ;;
esac

vic_weight=0.0
sig_weight=0.0
case "$regularizer" in
  none) ;;
  vic) vic_weight=1.0 ;;
  sig) sig_weight=0.01 ;;
  *) echo "unknown regularizer: $regularizer" >&2; exit 2 ;;
esac

case "$ldad" in
  off) decoder=false; ldad_weight=0.0 ;;
  on) decoder=true; ldad_weight=1.0 ;;
  *) echo "unknown LDAD switch: $ldad" >&2; exit 2 ;;
esac

name="dldad_${target}_${regularizer}_${ldad}"
.venv/bin/python scripts/train.py \
  +experiment=disc_observed_ldad_factorial \
  run_name="$name" device="$device" \
  model.state_target="$state_target" \
  model.observed_action_ldad="$decoder" \
  objective.vicreg.weight="$vic_weight" \
  objective.sigreg.weight="$sig_weight" \
  objective.observed_action_ldad.weight="$ldad_weight"

# The value head is intentionally untrained in this matrix.  Retain only the
# transition-identification fields (matching and RSA) from this audit.
.venv/bin/python scripts/audit_counterfactual.py \
  ckpt="runs/$name/best.pt" device="$device" n_episodes=30
touch "runs/$name/DONE"
rm -f "runs/$name/last.pt"
.venv/bin/python scripts/report_observed_ldad_factorial.py
