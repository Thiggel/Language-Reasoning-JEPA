# ALFWorld non-oracle admission gate

## Decision

Determine whether official text-only ALFWorld can be collected and replayed
through the same non-oracle dynamic catalogue used by the paper evaluator.
Keep this data decision separate from model quality.

## Observations before admission

- Geometry JEPA and recurrent sentence LM process gates completed from commit
  `2711922584ec5f8dcc4c694ea5c7fb3d9c01e5f9` with checkpoints and finite
  non-oracle metrics. Their two-epoch success is zero and is not scientific
  model evidence.
- Recurrent token LM on Alex and recurrent sentence-latent LM on Grete remain
  scheduler-pending. They are neither failed nor complete.
- Official ALFWorld train, seen-validation, and unseen-test engine smokes
  exactly replay 36 factual transitions. One recoverable alternative per
  transition also completes.
- The evaluator uses only observed entity names and a fixed action grammar.
  ALFWorld admissible commands and the expert are collection labels.
- Fast Downward mappings leak for the lifetime of a Python process. Per-episode
  process isolation is therefore a validity and storage requirement.

## Falsifiable next gate

Submit one bounded collection/replay job per ALFWorld split plus Grünau
recoveries for the two external pending process gates. Any expert catalogue
miss, replay drift, nonterminal expert trace, scratch-space accumulation,
missing checkpoint, non-finite metric, or evaluator oracle-menu dependency
blocks scale-up. Passing permits the remaining ALFWorld admission controls,
not the full learning-rate sweep.

## Human steering incorporated

The human explicitly asked to finish ALFWorld and submit all currently
admissible work. This authorizes the official dataset setup and bounded jobs,
but the frozen paper contract still forbids broad model selection before
process and per-dataset gates pass.

## Report

See [the self-contained implementation and admission report](../../reports/intent_phrase/2026-07-22-alfworld-admission/REPORT.md).
