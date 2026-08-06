# Sparse multidepth Endpoint-Energy factorial

## Question

Test whether recursively imagined endpoint Energy becomes more useful for
planning when trained with absolute distance calibration, within-root ranking,
dense rollout dynamics, temporal straightening, or monotonicity.

## Factorial

| Axis | Values |
|---|---|
| Root-to-endpoint depths | 0, 1, 2, 4, 8, 16 |
| Energy loss | MSE, ranking, MSE plus ranking |
| Rollout supervision | Energy-only, dense latent dynamics |
| Geometry | none, raw straightening, projected straightening, Energy monotonicity, projected-distance monotonicity |
| Planning | final endpoint Energy at depths 1, 2, 4, 8, 16 |

Depth `d` denotes the endpoint predicted from the same root after `d`
recursive predictor applications. Depth zero is the current-state anchor.
Ranking compares candidate continuations from the same root at the same
horizon. Monotonicity applies only where the EMA teacher confirms improvement.
Deeper evaluation uses symbolic future feasible-action menus and is therefore
candidate-privileged.

Every cell uses 300,000 generated training examples, batch size 16, seed zero,
300 evaluation episodes, and beam width eight. Each allocation sequentially
runs the three Energy losses. These are full-budget selection runs; selected
recipes require four additional seeds for a paper-level five-seed estimate.

## Implementation and validation

Commit `79d7fb378128` added sparse prefix selection, terminal-safe absorbing
padding, absolute endpoint-Energy regression, temporal straightening,
monotonicity objectives, runner scripts, and depth-16 evaluation. The model
test suite passed 64 tests. A CPU end-to-end smoke test exercised training and
multidepth evaluation.

The first submitted wave failed before training because the two new runner
scripts were excluded from immutable `git archive` snapshots. Commit
`29fe9aa1393d` added both scripts to `.gitattributes` and archive inspection
confirmed their presence. No GPU training time was consumed by that failure.

## Repaired placement

Six allocations were started on Grünau: all five Energy-only geometry variants
and the dense-dynamics/no-geometry control. The four remaining dense geometry
variants were submitted to Lise. One redundant projected-monotonicity Lise
submission was created when an interrupted controller command completed after
its recovery was issued; only one copy is scientifically required.

The autonomous controller watcher remained paused after manual submission.

