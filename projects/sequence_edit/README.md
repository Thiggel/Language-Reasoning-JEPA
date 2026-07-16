# Sequence-edit JEPA

This subproject models a mutable reasoning buffer. States encode the current
draft; actions are delete/insert/replace intents; planning repeatedly edits
the buffer until it matches a valid solution.

Primary implementation:

- `src/textjepa/data/edits/`
- `src/textjepa/models/edit_jepa.py`
- `configs/experiment/edit_*`
- `runs/edit_*`
- `reports/edit_jepa.tex`

The research index is
[`research/sequence_edit/`](../../research/sequence_edit/README.md). Historical
raw logs remain locally under `research/archive/edit_track/` until a new
sequence-edit cycle is opened. The track is a current scientific subproject
but has no active controller decision at present.
