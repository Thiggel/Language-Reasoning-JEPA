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

**Protocol correction:** the original compiled pilot exposed privileged
feasibility labels to a learned support head. That hybrid is discarded and
cannot support the paper claim. Environment collection/replay remains valid,
but model evidence must be regenerated with a full catalogue, no action or
feasibility prior, and rejected actions stored only as observed transitions.

The first transition-only rebuild completed with perfect replay and expert
catalogue recall, but six candidate games timed out under the 300-second
branch cap. The collector substituted later games, yielding 8 episodes and
115 factual / 230 counterfactual / 115 rejected-action transitions instead of
the predeclared 150 / 300 / 150. This is process-valid data but a failed
scientific admission gate. The recovery pins the exact original eight game
identities and raises only the per-episode cap; it may not substitute easier
games.

That exact-identity recovery also failed admission despite completing cleanly.
ALFWorld's hand-coded expert took 11 rather than 57 steps through one fixed
game, demonstrating that factual transition count is not stable enough to be
an identity criterion. More importantly, one of 104 states had no retained
rejected-action branch because the collector tried only its first shuffled
invalid candidate. The collector now retries up to the declared attempt
budget, and the validator independently requires exactly one admissible and
one rejected-action branch at every factual state. The 57-step game itself
exceeded a 15-minute strict-coverage smoke, so the bounded fixture freezes the
eight identities selected by the first pre-model collection that completed
full coverage. This exclusion is based on collection validity and elapsed
time, not model performance.

The corrected v3 gate passes. Its frozen template SHA256 is
`e9181dbecd1889071bad45ac3562f97b0d5fdd8fa5203e5fbf6fc571baf94c15`.
It contains 8 episodes, 115 realized factual transitions, 230 counterfactuals
with exactly 115 rejected-action outcomes, zero failed games, and independently
validated full per-state coverage. Exact replay, goal success, and expert
catalogue recall are all 100%. This admits the fixture for the bounded
learnability gate, not for headline performance claims.

## Falsifiable next gate

Compare prior-free JEPA+GAR, token LM, sentence LM, and sentence-latent LM on
the admitted pilot v2 using exactly the same full catalogue. Use approximately
matched optimizer-update counts and two learning-rate checks per family. Each
family must reach at least 75% strict train success before any full ALFWorld
collection or paper-scale sweep is admitted. Evaluate JEPA at one simulated
step first; deeper simulation is admitted only after basic learnability.

## Human steering incorporated

The human explicitly asked to finish ALFWorld and submit all currently
admissible work. This authorizes the official dataset setup and bounded jobs,
but the frozen paper contract still forbids broad model selection before
process and per-dataset gates pass.

## Report

See [the self-contained implementation and admission report](../../reports/intent_phrase/2026-07-22-alfworld-admission/REPORT.md).
