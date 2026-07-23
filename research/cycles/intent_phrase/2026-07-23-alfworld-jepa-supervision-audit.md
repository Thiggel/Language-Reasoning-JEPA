# Cycle: ALFWorld JEPA supervision audit

Status: decision complete; current no-prior ALFWorld result invalidated

## Decision

Do not interpret zero closed-loop training success as a JEPA memorization
failure and do not scale the current recipe. Repair causal-prefix consistency
and direct counterfactual state-latent supervision first.

## Falsifiable question

Does the model fail to fit stored state transitions, or does the
counterfactual training interface differ from the planner interface?

## Outcomes that changed the direction

- Expert transition error near persistence would have triggered an
  optimization/capacity audit.
- Healthy expert fit but poor observed-counterfactual fit would trigger a
  supervision repair.
- Healthy observed-candidate fit but poor full-catalogue rank would trigger a
  candidate-coverage experiment.

## Evidence

Across 115 stored states, both learning rates retrieve the exact expert target
state 100% of the time and reduce latent error from about 1.10 persistence to
0.327/0.246. Across 230 alternatives, planner-consistent error is 1.002/0.925
versus 1.019/0.995 persistence. The model fits 98.5%/97.5% of configured GAR
pairs, revealing that the configured labels—not optimization—are the immediate
problem. Code inspection shows omitted causal prefixes in both alternative
prediction and rollout-target encoding.

No human steering note was present in `.researchctl/steering/inbox/`.

## Next validity gate

Synthetic tests must prove that pre-anchor history affects executed and
alternative prediction identically and is included in all horizon targets.
The repaired eight-game model must then beat persistence on observed
counterfactual states before full-catalogue success is interpreted.

Full explanation:
[`2026-07-23 report`](../../reports/intent_phrase/2026-07-23-alfworld-jepa-supervision-audit/REPORT.md).
