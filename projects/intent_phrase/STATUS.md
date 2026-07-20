# Intent-phrase project status

_Updated 2026-07-20._

The latest terminal audit produced no new scientific metrics. Seven GAR jobs
timed out before optimization due to an overlong multiprocessing socket path;
two external-cluster counterfactual jobs failed because their exact snapshot
lacked the requested configuration; seed 0 is not terminal. Current work is
still unresolved, so no new round is planned from this audit.

An evening steering update authorizes one minimal recovery: rerun only the
four-step geometry-teacher seed with `TMPDIR=/tmp`, compare it with the
existing matched two-step seed, and leave the active Alex counterfactual job
untouched. This is a pending test, not new evidence.

The subsequent three-seed J3 checkpoint audit supersedes that pending
recovery. Teacher top-1 is `.890 +/- .036` and student-versus-oracle top-1 is
`.907 +/- .006`, while strict success remains `.588 +/- .013`. State variance
and effective rank are healthy, but task-value decodability falls from 1.000
in observed states to `.380 +/- .014` after one prediction and `.263 +/- .004`
under recursive rollout. The next decision is therefore a matched causal-
context and learning-rate screen, not deeper teacher lookahead.

The first four-cell diagnostic launch is infrastructure-invalid: the Grünau
controller overwrote plan-level `TMPDIR=/tmp` with its long run-directory
temporary path, and all four jobs reproduced the Unix-socket path failure
before optimization. A v2 plan uses the repository's post-controller short-
path wrapper and waits for those invalid slots to terminate or be explicitly
cancelled.

Valid short-path runs now settle the seed-0 screen. Full-history `1e-3`
reaches `.705` strict and `.920` slack-two versus `.590`/`.890` for the matched
`3e-4` seed, with healthy state rank and variance. Context 1 (`.155`), context
4 (`.475`), and full-history `1e-4` (`.295`) regress. The next decision is a
two-seed confirmation of `1e-3`, not another context or rate sweep.

## Defensible observations

| Observation | Status |
|---|---|
| The action-conditioned model can match counterfactual next-state transitions at approximately 0.99 in grounded configurations. | supported in the stylized environment |
| Non-symbolic two-step latent-goal preference distillation produces the dominant causal-model gain. | supported by the causal build-up |
| The causal-transformer JEPA currently trails the matched token intent policy. | supported; central open problem |
| Direct next-state prediction is substantially better than residual prediction in the current causal matrix. | supported |
| Dense rollout depth four does not improve control and increases seed variance. | supported for the current recipe |
| LDAD and monotonicity improve strict success separately. | provisional until combined/matched confirmation |
| Hierarchical planning improves this project. | contradicted by corrected confirmations |
| A supervised action prior improves deployable one-step planning. | supported across three seeds through length 9 |
| JEPA reranking adds value after prior proposal. | contradicted in the current top-2/top-4 evaluation |
| Deep latent rollout improves planning. | promising only with oracle future-action menus; not deployable evidence |
| Learned-catalogue depth-two/four planning works without symbolic menus. | implementation validated; scientific pilot pending |
| Explicit causal action history improves catalogue availability. | supported one-seed mechanism result; length-nine validity gate still fails |
| Pure JEPA scoring should discard the learned proposal score after top-four construction. | contradicted by endpoint results; calibrated hybrid test pending |
| JEPA currently beats the matched token intent policy on the easier stylized domain. | contradicted by the compact comparison: .797 versus .827 strict success |

## Current causal validation matrix

| Condition | Strict | +2 actions |
|---|---:|---:|
| one-step latent dynamics | .098 +/- .045 | .495 +/- .115 |
| + observed-outcome prediction | .123 +/- .076 | .500 +/- .109 |
| + recursive outcome consistency | .125 +/- .072 | .483 +/- .101 |
| + two-step latent-goal preference distillation | .588 +/- .013 | .845 +/- .043 |
| + dense rollout depth 4 | .510 +/- .106 | .812 +/- .062 |
| + faithful action-displacement decoding | .632 +/- .040 | .868 +/- .020 |
| + scalar value distillation | .602 +/- .041 | .872 +/- .028 |
| + terminal-distance monotonicity | .637 +/- .008 | .848 +/- .038 |

These are validation results, not a sealed final-test table. The current
causal reference is not yet strong enough to support the desired headline.

## Blocking uncertainties

1. Is the gap caused by causal-predictor optimization/capacity, or by the
   learned preference student failing to reproduce a reasonably strong
   latent-goal teacher?
2. Do LDAD, monotonicity, value calibration, and observed counterfactuals
   combine constructively, or are their individual gains redundant?
3. Does the selected stylized recipe transfer to faithful iGSM under shuffled
   action menus?
4. Does any gain survive parameter/compute-matched LM tuning and a sealed
   final test?
5. Which representation properties causally predict robust action selection,
   rather than merely being linearly decodable?
6. Does retaining calibrated proposal evidence make JEPA consequence
   simulation useful, or does the current value model add no deployable
   ordering signal?
