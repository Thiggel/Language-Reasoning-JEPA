# ICLR 2027 intent-phrase paper draft

This directory contains the anonymous paper draft for the observed
intent-phrase subproject. It uses the official ICLR 2027 style files downloaded
from <https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip>.
The style files are copied without modification.

Build from this directory:

```bash
bash make_figures.sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The two conceptual figures are native TikZ sources. Empirical claims use
tables until an analysis has a meaningful quantitative axis and a validated
source artifact. The old bar charts, preliminary length plot, invalid scaling
curve, and blank minimal-pair panel remain as historical files but are not
included in the manuscript.

Evidence tiers used in the manuscript:

1. Five-seed results are eligible for headline claims.
2. One-seed mechanism controls are labeled exploratory.
3. Candidate-privileged and oracle diagnostics are labeled in text and
   captions.
4. Missing experiments remain `pending` rather than receiving inferred values.

The controlled paraphrase, negation, and operator-swap analysis must retain its
full-space statistics and frozen feature source before its shared t-SNE and
UMAP illustrations enter the paper. The current scaling pilots are excluded.
