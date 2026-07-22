# ALFWorld tiny-set learnability gate

## Decision

Determine whether a small geometry JEPA can memorize the validated eight-episode
ALFWorld pilot before collecting or sweeping a larger dataset.

## Result

The oracle solves every train and seen-validation episode and random solves
none. Three matched JEPA runs at learning rates 3e-4, 1e-3, and 3e-3 solve
zero of eight train episodes at both strict and +4 budgets. The two higher
rates reduce invalid actions to at most 0.7%, showing that feasibility is
learned, but goal-directed action choice is not.

The first submissions were invalid infrastructure runs: a procedural
fresh-epoch sampler indexed beyond the fixed episode list at epoch one. The
repaired runs repeat fixed examples and complete all 300 epochs.

## Falsifiable next gate

Run a candidate-privileged, teacher-forced localization audit on all 150 train
states. Measure expert-action recall in support top-M, prior-only choice, and
one- and two-step JEPA reranking. If expert recall is low, repair proposal
training; if recall is high but reranking is wrong, densify geometry/value
supervision. Do not collect more ALFWorld data until this distinction is made.

## Report

See [the self-contained result report](../../reports/intent_phrase/2026-07-22-alfworld-overfit-failure/REPORT.md).
