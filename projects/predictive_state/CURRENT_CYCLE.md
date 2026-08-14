# Current cycle

Stage 3A, `2026-08-14-qwen-stage3a-oracle-geometry-v1`.

## In flight

Update this table on every launch and every completion. A cell that is running
and not listed here has not been handed off.

| Started | Round / cell | Where | What it decides |
|---|---|---|---|
| 2026-08-14 | `2026-08-14-qwen-stage3a-oracle-geometry-v1` state collection + oracle geometry probe | gruenau11:2 | whether terminal-state distance is monotone along correct GSM8K solutions, against four controls |

Nothing else is running. Completed rounds move to `EXPERIMENT_INDEX.md` and
their numbers to `STATUS.md` or a dated report.

## Where the project stands

Stage 1 is closed as a small, nearly free, real effect that does not meet its
gate, on fifteen token-matched cells. Stage 2's efficiency half passes
untrained (1.86x throughput, cache growth halved) while its fidelity half fails
by roughly 4x, and the first curriculum was under-budgeted at about 7M tokens
against the protocol's 60M. Stage 3A has 4,800 verified GSM8K trajectories and
is being measured now. Numbers are in `STATUS.md`; the full account is in
`research/reports/predictive_state/2026-08-14-stage1-closed-stage2-started/`.

## Direction-changing outcomes for 3A

- Distance to an oracle terminal falls monotonically along correct solutions and
  separates correct from incorrect at matched prefixes, and the four controls
  (another correct trajectory for the same problem, a different problem with the
  same answer, a random terminal, the same relative position elsewhere) do not:
  the geometry exists and a learned energy head is worth building.
- Monotonicity holds but the controls also score well: the signal is position or
  length, not goal direction. Report as a negative and do not build on it.
- Monotonicity fails on a model that solves only 29.6% of problems: rerun on a
  stronger checkpoint before concluding anything about the geometry itself.

## Next, in priority order

1. Stage 2 at the protocol's full 60M-token budget, so the current negative is
   not attributable to my sizing.
2. Speculative-verification evaluation. Top-20 agreement near 0.90 with top-1
   near 0.52 is exactly the drafting profile, verification makes it lossless,
   and it needs no training. Probably the strongest available Stage 2 claim.
3. Rewrite Stage 3B before implementing it. See the normative conflict below.
4. Owed controls: frozen backbone at matched tokens, persistence baseline, and
   a long `ntp_weight = 0` cell.

## Normative conflict blocking Stage 3B

The 2026-08-14 rule in `CLAUDE.md` puts steps-to-go regression and symbolic
ranking labels out of scope, even as diagnostics-turned-components. Stage 3B in
`DESIGN.md` trains the goal distance with Huber regression to remaining chunks
and a margin driven by verified correct/incorrect terminals, and trains a direct
value head on verified outcomes. Both are now out of scope as components.

Stage 3A is unaffected: it is a labeled oracle measurement, not a component.

Stage 3B must be rewritten so the energy head is trained by ranking
counterfactual continuations against the one that actually occurred, which is
self-supervised because the observed continuation is in the data. The verifier
stays evaluation-only. `DESIGN.md` has not been rewritten yet and currently
describes the superseded design.
