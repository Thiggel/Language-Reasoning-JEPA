# Intent-phrase project status

_Updated 2026-07-16._

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
