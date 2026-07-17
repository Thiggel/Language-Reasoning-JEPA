# Sequence-edit JEPA research records

The sequence-edit project models a mutable reasoning buffer and plans
delete/insert/replace intents until the draft is correct.

Current artifacts:

- implementation: `src/textjepa/data/edits/` and
  `src/textjepa/models/edit_jepa.py`;
- experiment configurations: `configs/experiment/edit_*`;
- run families: `runs/edit_*`;
- scientific deck: `reports/edit_jepa.tex` and `reports/edit_jepa.pdf`.

Historical console logs remain locally under `research/archive/edit_track/`.
They are intentionally not duplicated here. A new active experiment cycle
should add `waves/`, `logs/README.md`, and `BACKLOG.md` under this directory.

Active cycle: token-aligned recursive edit-state restart, documented in
[`../cycles/sequence_edit/2026-07-17-structured-token-edit-restart.md`](../cycles/sequence_edit/2026-07-17-structured-token-edit-restart.md).
