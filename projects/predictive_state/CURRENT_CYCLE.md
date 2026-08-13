# Current cycle

`2026-08-12-qwen-stage1-pressure-v1` (complete)

Stage 1's representation claim is closed as a negative, and the evidence for
Stage 2 is better than it was. Full numbers in `EVIDENCE.md`.

Across the screen and the pressure round, fifteen token-matched 20M-token cells
now agree that the auxiliary objective does not shape this backbone. Every one
lands within 0.0021 nats of the NTP-only control, across a 30x range of
prediction weight, a 56x narrowing of the state channel, a linear-only
predictor, and single-state conditioning. The cross-model probe says the same
independently: at layer 24 ordinary training moves the representation about 5%
of CKA away from the original checkpoint and the objective adds 0.02% on top.

The decisive cell is `proj8`. It genuinely crippled the predictor, more than
doubling direction error, and the backbone did not compensate. The mechanism we
hoped for, that a predictor unable to solve the transition would force the
states to become more predictable, does not operate here.

Three findings do carry forward to recurrent execution:

- the transition is close to linear, cosine 0.1335 for a full-rank linear map
  against 0.1067 for the SwiGLU;
- eight dimensions of state, with the action channel intact, reach 0.1508
  against 0.2078 for action-only, so the state's contribution is low
  dimensional;
- layer 24 is redundant. Layer 18 alone reaches 0.1050, marginally better than
  both sources, while layer 24 alone reaches 0.1252.

Together with the screen's calibrated activation scale (RMS ratio 1.0006) these
say the predicted state is cheap, accurate and injectable, which is exactly what
Stage 2 needs and what Stage 1 failed to deliver on its own terms.

Next, in order:

1. Recommend to the project owner that the paper's weight moves to Stage 2.
   Stage 1 becomes a reported negative with the pressure round as its evidence,
   which is a stronger and more honest section than a marginal positive.
2. Frozen-backbone `full` at 20M tokens. Still owed: the cosine improvement over
   the frozen diagnostic confounds adaptation with 200x more optimization, and
   this single cell separates them. Cheap and it closes an obvious hole.
3. Persistence baseline on these checkpoints, against the 0.258 unrelated-pair
   cosine floor.
4. Before any Stage 2 run, revisit the source split. `src18only` and
   `proj8-action448` together suggest the recurrent path may only need a
   narrow slice of one layer.

Open items this cycle does not settle:

1. One seed throughout, as the screen protocol specifies.
2. Qwen2.5-0.5B only. The negative is established at 0.5B; whether it holds at
   OLMo 1B is untested, though nothing in the pattern suggests scale is the
   binding constraint.
3. Outlier-robust scale reporting: `predicted_rms`/`target_rms` are linear means
   over a heavy-tailed distribution while the objective is a log-space Huber.
