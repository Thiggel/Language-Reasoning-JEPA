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

## Result

The bounded pilot contains 8 train, 4 seen-validation, and 4 unseen-validation
episodes (291 factual transitions and 291 recoverable counterfactuals). The
split identities are disjoint. Every retained episode exactly replays, reaches
its goal, and retains every expert action in the non-oracle catalogue. One
pathological seen-validation game exceeded the five-minute per-game bound and
was recorded and skipped rather than silently hanging the dataset build.

All four model process pathways also completed on Grünau. External queued
duplicates are obsolete and do not affect this decision.

## Falsifiable next gate

On this fixed admission-only pilot, compare the real-engine random and
privileged expert-replay bounds and train the small geometry JEPA at three
log-spaced learning rates. The model must substantially overfit train strict
success before any full ALFWorld collection or paper-scale sweep is admitted.
If it cannot, diagnose optimization and action-support supervision rather than
spending compute on scale.

## Human steering incorporated

The human explicitly asked to finish ALFWorld and submit all currently
admissible work. This authorizes the official dataset setup and bounded jobs,
but the frozen paper contract still forbids broad model selection before
process and per-dataset gates pass.

## Report

See [the self-contained implementation and admission report](../../reports/intent_phrase/2026-07-22-alfworld-admission/REPORT.md).
