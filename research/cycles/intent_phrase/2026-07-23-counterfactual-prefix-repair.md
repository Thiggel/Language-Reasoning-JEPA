# Counterfactual causal-prefix repair

## Decision

Do not admit paper-scale ALFWorld training yet. Test whether broader exact
observed-counterfactual coverage improves full-catalogue ranking and
closed-loop control.

## Falsifiable question

After both causal-prefix paths are repaired, does direct full-state
counterfactual latent prediction beat persistence on the fixed eight-game
memorization gate?

## Result

Yes for observed alternatives: weight-one models reduce alternative error by
60--68% relative to persistence at learning rates 0.001 and 0.003. The
weight-zero control stays near persistence. Factual next-state retrieval and
stored-candidate GAR fitting remain high.

No for deployable planning: all completed cells solve zero training episodes;
full-catalogue expert top-1 is 11--12%, despite 80--86% top-1 on the
candidate-privileged stored-alternative set.

## Validity

Three cells completed at exact snapshot
`a714b43dd2bbe38e8f8664e779c57ccaa99e16c9`. The weight-zero,
learning-rate-0.003 cell is excluded after a stale NFS handle in TensorBoard.
There are 115 factual states and 230 observed counterfactuals. This is a
one-seed memorization gate, not a generalization result.

## Next gate

Increase exact admissible and rejected-action branch coverage on the same
games. Scale only if held-out transition error, full-catalogue expert rank,
chosen-action feasibility, and closed-loop success improve together.

Report:
[`2026-07-23-counterfactual-prefix-repair`](../../reports/intent_phrase/2026-07-23-counterfactual-prefix-repair/REPORT.md)
