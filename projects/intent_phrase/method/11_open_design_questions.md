# Open decisions

| Priority | Question | Smallest decisive test |
|---:|---|---|
| 1 | Does scaling survive without feasible menus? | Existing checkpoint, full catalogue at every node |
| 2 | Can training learn invalid actions? | One full-catalogue training seed |
| 3 | Can depth-1 improve simply? | Fixed-H4 root distillation |
| 4 | Is 0.25 auxiliary weight optimal? | `0,0.1,0.25,0.5,1` one-seed screen |
| 5 | Does prefix endpoint ranking help? | rank endpoints at all rollout prefixes |
| 6 | Is recursive prediction necessary? | direct sequence ranker |
| 7 | Does a causal predictor reproduce scaling? | matched causal checkpoint |
| 8 | Is width or depth the useful compute? | depth × beam-width FLOP curve |
| 9 | Does the result generalize? | BlocksWorld, graph navigation, logic planner |
| 10 | Can proposals be learned? | learned top-M retrieval then JEPA reranking |

## Full-catalogue decision rule

- Scaling survives immediately: replicate existing checkpoints over five seeds.
- Scaling fails but invalid rate falls after full-catalogue training: use the
  trained full-catalogue method.
- Invalid rate remains high: separate proposal/feasibility learning is required
  and must be reported as an additional component.

## One-step improvement order

1. Remove the pair-difference auxiliary.
2. Sweep its weight.
3. Distill fixed H4 quality into a dedicated first-action head.
4. Rank every rollout prefix.
5. Add a learned mixture of first-action and endpoint Energies.

Do not complicate the primary method unless an improvement is reproducible and
does not weaken deep scaling.
