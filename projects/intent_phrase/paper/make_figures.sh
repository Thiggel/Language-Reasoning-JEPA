#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
REPORT="$ROOT/research/reports/intent_phrase/2026-08-04-current-findings"
cp "$REPORT"/figures/headline_models.pdf "$HERE/figures/"
cp "$REPORT"/figures/gar_mechanism.pdf "$HERE/figures/"
cp "$REPORT"/figures/representation_diagnostics.pdf "$HERE/figures/"
cp "$REPORT"/figures/length_ood.pdf "$HERE/figures/"
cp "$REPORT"/report_data.json "$HERE/data/results.json"
cd "$HERE/figures"
pdflatex -interaction=batchmode -halt-on-error test_time_scaling_repaired.tex
pdflatex -interaction=batchmode -halt-on-error minimal_pair_geometry_pending.tex
pdflatex -interaction=batchmode -halt-on-error headline_models_clean.tex
mv headline_models_clean.pdf headline_models.pdf
rm -f ./*.aux ./*.log
