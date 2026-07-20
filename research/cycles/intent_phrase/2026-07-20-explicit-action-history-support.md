# Cycle: explicit action-history support

Status: implementation validated; bounded causal pilot planned

## Decision

Determine whether learned-catalogue feasibility fails because a pairwise MLP
cannot recover prerequisite identity from one frozen JEPA state, or because
the action catalogue itself is insufficient.

## Falsifiable comparison

The aligned model lets each candidate intent phrase attend to the lexical
embeddings of intents already executed or imagined. The matched negative
control uses the identical attention head, parameters, training examples, and
loss while masking the entire history. Both retain the same frozen JEPA state,
predictor, value function, catalogue, prior supervision, and planner budget.

Run aligned history attention at learning rates 1e-3 and 3e-3, plus the
no-history control at 3e-3. Evaluate learned support weights 1, 3, and 10 at
depths one, two, and four with matched prior-only and JEPA scoring.

## Result patterns

- If aligned history reduces length-nine invalid episodes below .25 while the
  no-history control remains poor, retain explicit history as the proposal
  interface and then interpret the JEPA depth curve.
- If aligned and no-history both improve similarly, the larger lexical head,
  not history, is sufficient; simplify to the control architecture.
- If neither improves, test a token-level prerequisite parser or policy LM;
  do not continue coefficient or head-only learning-rate sweeps.
- If only length six improves, train/evaluate a length-balanced curriculum
  before changing width.

## Validity gates

No symbolic feasibility query or oracle future menu enters inference. Training
feasibility labels remain explicit supervision and are disclosed. Prefix masks
are strictly causal; the current candidate and future actions are excluded.
Imagined action prefixes, not true future actions, condition deeper support.
The no-history control differs by one mask only. Frozen JEPA components receive
no gradient. Invalid selected actions remain failures without fallback.

## Implementation evidence

Regression tests cover strict-prefix masks, the no-history control, empty
initial history, attention gradients, and propagation of imagined histories
at depth two. A checkpoint-initialized smoke resets only the new availability
head, trains 0.80M parameters with finite loss, and completes non-oracle
depth-two evaluation. The full repository suite reports 124 passing tests.

Human-facing implementation report:
[`../../reports/intent_phrase/2026-07-20-explicit-action-history-support/REPORT.md`](../../reports/intent_phrase/2026-07-20-explicit-action-history-support/REPORT.md).

