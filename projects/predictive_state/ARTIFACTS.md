# Artifact map

- Scientific contract: `projects/predictive_state/`.
- Model code: `src/textjepa/models/action_transition.py`.
- Objectives: `src/textjepa/objectives/predictive_state.py`.
- Frozen token blocks and trajectory schemas: `src/textjepa/data/predictive_state.py`.
- Upper-stack recurrence and training helpers: `src/textjepa/training/predictive_state.py`.
- Representation/geometry metrics: `src/textjepa/analysis/predictive_state.py`.
- Goal models and block search: `src/textjepa/planning/predictive_state.py`.
- Recipes: `configs/predictive_state/`.
- CLIs: `scripts/*predictive_state*` and `scripts/*action_transition*`.
- Runs: `runs/autonomy/predictive_state/`.
- Shared corpora and their digests: `runs/autonomy/predictive_state/_corpora/`
  (WikiText-103 and FineWeb-Edu token blocks, GSM8K problems). Every cell in
  a comparison reads one pre-built file so tensors are identical by
  construction rather than by matching digests afterwards.
- Dated reports: `research/reports/predictive_state/<date>-<topic>/REPORT.md`.

Historical paths for the other three subprojects remain unchanged. Shared code
does not make results from those projects evidence for this one.
