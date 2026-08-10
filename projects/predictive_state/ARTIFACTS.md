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

Historical paths for the other three subprojects remain unchanged. Shared code
does not make results from those projects evidence for this one.
