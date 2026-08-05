# Baselines, ablations, and controls

## Why the candidate interface is held fixed

The main experiment asks whether different learning objectives can value the
same candidate language actions. Supplying the same current feasible menu
removes proposal coverage as a confound.

This does not claim a complete language agent. It is a laboratory intervention
that isolates action consequence evaluation.

## Random policy

Uniformly choose among currently feasible actions. This establishes how much
success comes from the environment structure and action budget alone.

## Token language model

An autoregressive token model scores candidate action phrases from the prompt
and reasoning history. It predicts language tokens rather than a successor
latent state.

Candidate score should use normalized token likelihood so longer wording is not
penalized merely for containing more tokens.

## Sentence language model

A sentence-level generative model predicts the next intent or outcome unit at
sentence granularity. It is a stronger comparison when the action candidates
are complete intent phrases.

## Sentence model plus latent loss

This hybrid combines a generative sentence objective with latent consequence
prediction. It tests whether the objectives are complementary rather than
forcing a JEPA-versus-language-model contest.

## Next-sentence latent-MSE model

This model predicts the next sentence representation using latent MSE but does
not use GAR. It is the closest test of whether latent prediction alone is
sufficient for planning.

## Looped models

Looped token and sentence models reuse a Transformer block a variable number of
times. Training samples the loop count, for example from a Poisson-like
distribution, and evaluation varies loops to measure test-time compute.

Comparison with JEPA should use measured or defensibly estimated FLOPs, not raw
loop count versus beam depth.

## MSE-only JEPA

Train successor prediction without local action ranking. This tests the central
claim that prediction does not identify planning geometry.

## Counterfactual-only JEPA

Supply the same alternative successor examples and counterfactual latent loss,
but remove preference ranking. This tests whether GAR's gain comes merely from
additional counterfactual data.

## No geometric ranking

Keep the architecture and other losses but set the GAR ranking weight to zero.
This differs from a pure MSE-only model when other Energy or counterfactual
terms remain active. Configurations should state exactly which losses survive.

## Direct ranker

Score `(state, action)` directly with the same positive and counterfactual
negative examples. It controls for pairwise behavior cloning without predictive
factorization.

## True-state Energy

Train the Energy head on EMA-encoded true successors rather than online
predicted successors. It tests robustness to prediction error and whether GAR
gradients need to shape the predictor.

## State Energy versus transition Energy

- State Energy predicts absolute quality from the terminal imagined state.
- Transition Energy scores the relation between the predecessor and successor.

State Energy naturally composes through terminal beam scoring. Transition
Energy must have a declared composition rule; terminal-only is now the default.

## Exact and oracle controls

- **Exact-transition encoded Energy:** execute hypothetical actions exactly,
  encode their true outcomes, then apply the learned Energy.
- **Oracle-goal distance:** score an imagined or exact endpoint by distance to
  an encoded solved trajectory.
- **Exact symbolic distance:** use exact remaining necessary steps.

Together these localize failure among candidate search, transition prediction,
representation geometry, and learned scoring.

## Fairness requirements

- Tune learning rate separately for each model family and dataset.
- State the number of tried hyperparameters.
- Use the same generated evaluation puzzles.
- Hold candidate menus fixed in headline comparisons.
- Match width/depth approximately where useful, but prioritize an objective
  swap with the same backbone over exhaustive parameter matching.
- Use five seeds only after a recipe passes one-seed validity and scale gates.

## What should not be combined in one headline number

- feasible-menu and full-catalogue planning;
- current-menu and future candidate-privileged planning;
- stylized and faithful iGSM;
- MLP and causal-Transformer JEPA;
- validation-selected and held-out test results;
- exact/oracle diagnostics and learned deployment scores.
