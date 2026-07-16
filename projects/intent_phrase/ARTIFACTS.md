# Intent-phrase project artifact map

Existing paths are preserved so historical checkpoints and generated reports
remain reproducible.

## Research records

- `research/intent_phrase/waves/`: one document per experiment wave.
- `research/intent_phrase/logs/`: raw-log index and historical logs.
- `research/intent_phrase/PAPER_PLAN.md`: paper-facing protocol and tables.
- `research/intent_phrase/BACKLOG.md`: historical staged backlog.

## Configuration ownership

- `configs/experiment/disc_*`: stylized observed-intent JEPA development.
- `configs/experiment/paper_causal_*`: causal-transformer paper matrix.
- `configs/experiment/real_*`: faithful iGSM transfer.
- `configs/experiment/lm_intent*`, `sentlm_intent*`, and
  `paper_{token,sentence}*`: information-matched language-model baselines.
- `configs/data/igsm.yaml` and `configs/data/igsm_real.yaml`: stylized and
  faithful data interfaces.

## Implementation ownership

- `src/textjepa/data/igsm/`: problem generator, environment, rendering, and
  candidate construction.
- `src/textjepa/models/discourse_jepa.py`: observed-action JEPA.
- `src/textjepa/models/core.py`: causal action-conditioned latent dynamics.
- `src/textjepa/objectives/`: transition, preference, displacement,
  counterfactual, and regularization objectives.
- `src/textjepa/planning/search.py` and `faithful_search.py`: flat deployed
  planners and faithful evaluation.
- `scripts/train.py`, `probe.py`, `plan.py`, and `eval_run.sh`: primary entry
  points.

## Run families

- `runs/disc_*`: stylized development and historical ablations.
- `runs/paper_causal_*`: common-protocol causal paper runs.
- `runs/real_*`: faithful iGSM transfer.
- `runs/lm_*` and `runs/sentlm_*`: matched policy baselines.

## Reports

- `reports/discourse_jepa_neurips.tex`: current scientific slide deck.
- `research/reports/intent_phrase/`: future self-contained cycle reports.

New intent-phrase experiments should use an `intent_phrase_` or
`paper_causal_` round/run prefix and link their machine-readable results from
the project status rather than adding uncatalogued top-level notes.
