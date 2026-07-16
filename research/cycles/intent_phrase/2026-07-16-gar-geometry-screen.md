# Cycle: geometric action-ranking horizon and candidate screen

Status: retry plan finalized; first launch infrastructure-invalid

## Decision

Determine whether the current causal intent-phrase model is limited by the
depth or candidate coverage of its JEPA-geometry teacher before combining
additional objectives.

## Falsifiable question

Holding the causal architecture, data, losses, and evaluation fixed, does
changing geometric lookahead `H` or the number of root alternatives `K`
improve strict closed-loop success relative to the existing `H=2, K=2` J3
reference?

The teacher uses EMA-encoded true counterfactual outcomes and terminal-state
distance. It does not use symbolic remaining-step, ancestor, or relevance
labels. Environment feasibility and rendered outcomes remain privileged
training interaction and must be disclosed as such.

## Pilot

Run one seed for `H={1,4,8,16}` at `K=2` and `K={1,4,8}` at `H=2`. The
existing three-seed J3 row supplies `H=2, K=2`. Every job also emits the GAR
teacher audit, prediction probes, rollout drift, and strict/slack planning
metrics.

Primary metric: strict closed-loop success. Secondary metrics: slack-2
success, teacher-versus-oracle top-1/pair accuracy, student-versus-teacher
top-1/pair accuracy, transition match, and recursive drift.

Validity gates:

- identical shuffled action menus and validation examples;
- no symbolic ranking objective;
- non-collapsed state variance/effective rank;
- GAR audit contains at least 100 anchors and finite teacher labels;
- action-shuffle and transition checks remain interpretable.

## Decision rule

- Advance a setting to two additional seeds only if it improves strict success
  by at least 0.05 over its matched seed or materially improves teacher quality
  without degrading student alignment.
- If teacher quality improves but student alignment does not, tune the
  preference student rather than increasing horizon further.
- If neither teacher quality nor strict success improves, retain `H=2, K=2`
  and move to the staged LDAD/monotonicity/value combination screen.

No unread steering note was present. The user's explicit instruction to queue
all presently useful jobs caused this one-seed bounded screen to be submitted;
paper-grade confirmation remains gated on its result.

The first Grünau launch reached the data loader but its controller-generated
temporary path exceeded the Unix-socket length limit. All seven processes
stalled before the first optimizer step and produced no scientific evidence.
The finalized v2 plan runs the identical immutable commands with child
`TMPDIR=/tmp`; submission waits for the invalid controller slots to terminate
or for explicit cancellation.
