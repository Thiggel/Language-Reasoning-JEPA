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

The headline, GAR, representation, and preliminary length figures are copied
from the current machine-readable evidence report. The repaired test-time
compute figure is generated from the latest leakage-free fixed-checkpoint
gate. The minimal-pair figure is an intentionally blank draft panel. It must
not be interpreted as a result.

Evidence tiers used in the manuscript:

1. Five-seed results are eligible for headline claims.
2. One-seed mechanism controls are labeled exploratory.
3. Candidate-privileged and oracle diagnostics are labeled in text and
   captions.
4. Missing experiments remain `pending` rather than receiving inferred values.

The completed controlled paraphrase, negation, and operator-swap analysis
should replace `figures/minimal_pair_geometry_pending.pdf`. The old
terminal-length-leaking planning curve is excluded from positive evidence.
