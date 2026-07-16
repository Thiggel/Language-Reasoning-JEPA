# TextJEPA subprojects

The repository has three scientific subprojects. They share reusable code in
`src/textjepa/`, but claims, protocols, experiment records, and paper plans are
kept separate.

| Subproject | Scientific question | Project home |
|---|---|---|
| Observed intent-phrase JEPA | Can an action-conditioned latent world model learn counterfactual reasoning dynamics and non-symbolic action selection? | [`intent_phrase/`](intent_phrase/README.md) |
| Token-level causal iGSM JEPA | Can token-to-phrase-to-sentence predictive hierarchy induce useful abstraction and support language generation/planning? | [`token_igsm/`](token_igsm/README.md) |
| Sequence-edit JEPA | Can a latent world model plan edits over a mutable reasoning buffer? | [`sequence_edit/`](sequence_edit/README.md) |

## Ownership convention

- `projects/<name>/` is the stable entry point for scope, status, artifacts,
  and next decisions.
- `research/<track>/` stores detailed experiment waves, raw-log indexes, and
  historical evidence. Existing paths remain intact for reproducibility.
- `configs/`, `scripts/`, and `src/textjepa/` are shared implementation trees.
  Each project manifest lists the files and naming patterns it owns.
- `runs/` remains append-only experiment storage. Run names are not moved,
  because checkpoints and reports contain those paths.

Results from one subproject are not evidence for another unless a transfer
experiment explicitly tests that claim.
