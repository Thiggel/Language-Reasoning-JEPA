# Intent-phrase project status

_Updated 2026-07-21._

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
| Token-level executed-intent history repairs most learned-catalogue feasibility failures. | supported one-seed pilot; prior-only, not a JEPA simulation claim |

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
6. Once token-level proposal support removes most invalid actions, does JEPA
   reranking improve over the learned prior, and at what simulation depth?
