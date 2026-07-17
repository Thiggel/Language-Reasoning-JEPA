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

No further K, data, hierarchy, rollout, or LDAD experiments are active. A
future restart must first introduce a token-aligned recursive state and pass a
CPU-scale causal fixture.

Current entry points:

- [`STATUS.md`](STATUS.md)
- [`faithful-token cycle`](../../research/cycles/sequence_edit/2026-07-16-faithful-token-hierarchy.md)
- [`data and counterfactual pilot report`](../../research/reports/sequence_edit/2026-07-17-faithful-edit-data-counterfactual-pilot/REPORT.md)
