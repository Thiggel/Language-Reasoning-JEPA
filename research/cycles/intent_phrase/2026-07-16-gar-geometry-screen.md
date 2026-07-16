# Cycle: geometric action-ranking horizon and candidate screen

Status: infrastructure-invalid terminal audit complete; valid retry/current work unresolved

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

## Terminal audit and decision update (2026-07-16)

### Observations

- All seven newly terminal GAR summaries are `TIMEOUT` with exit code 124,
  empty metrics, and no declared artifacts. Their stderr traces show
  `OSError: AF_UNIX path too long` in the multiprocessing resource sharer.
  They never produced an optimization result and are process-invalid, not
  scientific failures.
- Two newly terminal counterfactual-outcome jobs (seeds 1 and 2) are `FAILED`
  with exit code 1, empty metrics, and no artifacts. Both immutable external
  snapshots at commit `cb1371f107fcebad82ca5b784585850077a13a71` lack the
  requested `paper_causal_a_cfout` Hydra configuration. Seed 0 has a manifest
  but no terminal summary. These are also process failures and contribute no
  counterfactual evidence.
- The supplied allocation snapshot has three active GPUs and one pending job
  for intent_phrase, with 3.0 project GPU-hours and only 0.9 global GPU-hours
  left in the weekly window. The snapshot does not map those allocations to
  run identifiers, so no completion is inferred from it.

### Steering note

The sole unhandled note asks whether the easier iGSM instantiation already
shows a JEPA advantage over language modeling and can therefore serve as a
paper result. It changed the decision in three ways:

1. It made the easy-domain matched comparison an explicit paper gate rather
   than an implicit background belief.
2. It prevents describing the current stylized evidence as a JEPA win: compact
   memory reports strict success of `.827 +/- .003` for the matched token
   intent policy versus `.797 +/- .008` for the older reduced non-symbolic
   JEPA; the current causal JEPA is lower still (`.588 +/- .013` for J3).
3. It raises regeneration of a sealed, machine-readable, information-matched
   easy-domain table ahead of any paper claim, but does not justify interrupting
   or duplicating the already unresolved GAR/current work.

### Decision

Do not interpret or scale either terminal round, do not claim that JEPA beats
the matched language-model policy on the easier domain, and do not admit a new
experiment from this cycle. Preserve the original GAR falsifiable decision
until a valid retry produces its predeclared metrics. The result patterns that
change direction remain: a horizon/candidate setting must improve matched-seed
strict success by at least 0.05 or improve teacher quality without degrading
student alignment; otherwise retain `H=2, K=2` and proceed to the staged
grounded-objective combination screen.

No `research/intent_phrase/NEXT_PLAN.json` is written. A new plan is not
decision-relevant while the current jobs are unresolved, it could duplicate a
writer, and the supplied global weekly allowance (0.9 GPU-hours) cannot support
the faithful training needed for the next comparison. Any later plan must use
schema version 2 and project `intent_phrase`.

Human-facing report:
`research/reports/intent_phrase/2026-07-16-terminal-run-validity-audit/REPORT.md`.
