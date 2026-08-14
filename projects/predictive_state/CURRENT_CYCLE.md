# Current cycle

Stage 3A, `2026-08-14-qwen-stage3a-oracle-geometry-v1`.

## In flight

Update this table on every launch and every completion. A cell that is running
and not listed here has not been handed off.

| Started | Round / cell | Where | What it decides |
|---|---|---|---|
| — | nothing running | — | — |

Completed 2026-08-14: `2026-08-14-qwen-stage2-refresh-sweep-v1` (refresh flat in
N; speculative 1.38x lossless at draft length 2) and
`2026-08-14-qwen-energy-head-v1` (energy transfers to correctness at AUC 0.673
but loses to likelihood at 0.726).

Nothing is running. `2026-08-14-qwen-stage3a-oracle-geometry-v1` completed on
2026-08-14. Completed rounds move to `EXPERIMENT_INDEX.md`, and their numbers
to `STATUS.md` or a dated report.

## Where the project stands

Stage 1 is closed as a small, nearly free, real effect that does not meet its
gate, on fifteen token-matched cells. Stage 2's efficiency half passes
untrained (1.86x throughput, cache growth halved) while its fidelity half fails
by roughly 4x, and the first curriculum was under-budgeted at about 7M tokens
against the protocol's 60M. Stage 3A is complete and negative: the oracle goal
geometry is generic progress rather than goal direction, and it cannot rank
correct above incorrect. Numbers are in `STATUS.md`; the full account is in
`research/reports/predictive_state/2026-08-14-stage1-closed-stage2-started/`.

## 3A outcome: the second pre-registered branch

Monotonicity held and the controls scored almost as well, which was the
pre-registered "report as a negative and do not build on it" branch. A random
terminal from another problem reaches Spearman 0.7257 against the true goal's
0.7663, and matched-prefix correct-versus-incorrect ranking is at or below
chance for every metric.

Before treating this as settled about the geometry rather than about this
model, the open question is whether it survives a checkpoint that solves more
than 29.6% of problems. That is the one branch not yet excluded.

## Next, in priority order

1. Rebuild the energy head with same-distribution negatives, for example gold
   steps drawn from other problems, so it cannot separate positives from
   negatives on provenance. This is the single most informative next run.
2. Repeat 3A on a stronger reasoning checkpoint. The geometry negative is
   currently confounded with a model that solves 29.6% of problems and yields
   only 40 matched-depth candidate pairs.
3. Stage 2 at the protocol's full 60M-token budget, so the current negative is
   not attributable to my sizing.
4. Speculative-verification evaluation is done; see STATUS. Top-20 agreement near 0.90 with top-1
   near 0.52 is exactly the drafting profile, verification makes it lossless,
   and it needs no training. Probably the strongest available Stage 2 claim.
5. Rewrite Stage 3B before implementing it. See the normative conflict below.
6. Owed controls: frozen backbone at matched tokens, persistence baseline, and
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
