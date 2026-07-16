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
and now has an active non-symbolic faithful-iGSM token-edit hierarchy pilot.

The new interface uses official solution sentences as the mutable buffer and
literal token insert/delete/replace operations as primitive actions. It does
not use the earlier symbolic defect-count ranking or oracle feasible edit
menu. See
[`2026-07-16-faithful-token-hierarchy.md`](../../research/cycles/sequence_edit/2026-07-16-faithful-token-hierarchy.md).
