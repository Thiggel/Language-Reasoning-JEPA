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
sequence-edit cycle is opened. The track remains a scientific subproject, but
its pooled/step-attention faithful-iGSM state family is retired after failing
causal action-use gates.

The new interface uses official solution sentences as the mutable buffer and
literal token insert/delete/replace operations as primitive actions. It does
not use the earlier symbolic defect-count ranking or oracle feasible edit
menu. It is nevertheless candidate-privileged oracle denoising: the gold
solution supplies corruption tokens and the exact inverse repair path.
Counterfactual outcomes are mechanically executed from the current buffer and
carry no target-relative quality label.

A token-aligned recursive restart is now implemented and has passed its
CPU-scale process fixtures. The first GPU decision crosses corruption regime
and stabilizer choice; hierarchy and planning remain gated on primitive causal
action use.

Current entry points:

- [`STATUS.md`](STATUS.md)
- [`structured-token restart`](../../research/cycles/sequence_edit/2026-07-17-structured-token-edit-restart.md)
- [`faithful-token cycle`](../../research/cycles/sequence_edit/2026-07-16-faithful-token-hierarchy.md)
- [`data and counterfactual pilot report`](../../research/reports/sequence_edit/2026-07-17-faithful-edit-data-counterfactual-pilot/REPORT.md)
- [`token-aligned VICReg screen report`](../../research/reports/sequence_edit/2026-07-17-structured-edit-vicreg-screen/REPORT.md)
- [`token-aligned LDAD screen report`](../../research/reports/sequence_edit/2026-07-17-structured-edit-ldad-screen/REPORT.md)
- [`structured token-edit 12-hour synthesis`](../../research/reports/sequence_edit/2026-07-18-structured-edit-12h-synthesis/REPORT.md)
- [`learning-rate sensitivity report`](../../research/reports/sequence_edit/2026-07-18-learning-rate-sensitivity/REPORT.md)
- [`executable MPC generation report`](../../research/reports/sequence_edit/2026-07-20-executable-mpc-generation/REPORT.md)
- [`executable MPC generation cycle`](../../research/cycles/sequence_edit/2026-07-20-multiscale-mpc-generation.md)
