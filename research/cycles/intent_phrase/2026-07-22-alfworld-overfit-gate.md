# ALFWorld tiny-set learnability gate

## Decision

Determine whether a small geometry JEPA can memorize the validated eight-episode
ALFWorld pilot before collecting or sweeping a larger dataset.

## Result

**Protocol correction:** these runs used the now-discarded learned feasibility
proposal before JEPA reranking. They are not evaluations of the paper-facing
full-catalogue JEPA+GAR method and cannot be compared to the final LM baselines.

The oracle solves every train and seen-validation episode and random solves
none. Three matched JEPA runs at learning rates 3e-4, 1e-3, and 3e-3 solve
zero of eight train episodes at both strict and +4 budgets. The two higher
rates reduce invalid actions to at most 0.7%, showing that feasibility is
learned, but goal-directed action choice is not.

The first submissions were invalid infrastructure runs: a procedural
fresh-epoch sampler indexed beyond the fixed episode list at epoch one. The
repaired runs repeat fixed examples and complete all 300 epochs.

## Falsifiable next gate

Replace the hybrid with an information-matched full-catalogue protocol. Every
method receives the same non-oracle catalogue; JEPA+GAR must rank actions only
through predicted latent consequences and goal geometry. Invalid actions enter
JEPA training as observed state transitions, never as feasibility labels.

## Report

See [the self-contained result report](../../reports/intent_phrase/2026-07-22-alfworld-overfit-failure/REPORT.md).
