# Endpoint Energy

This file replaces the older GAR method specification. GAR remains a control,
not the primary paper method.

## Training example

Choose one anchor state from a generated solution trajectory. Under the
validated feasible-menu recipe, use:

- the factual next action;
- two sampled alternative feasible actions;
- four sampled continuations beneath each root;
- a horizon sampled from `{1,2,4,8}`.

For every root and continuation:

1. Execute the action sequence in the training environment.
2. Encode its true final history with the EMA encoder.
3. Recursively imagine the same sequence with the online predictor.
4. Score the imagined endpoint with the horizon-conditioned Energy head.
5. Rank imagined endpoints using distances between true EMA endpoints and the
   EMA encoding of the solved training trajectory.

## Geometric target

```text
normalize(z) = layer_norm(z)
distance(z, goal) = mean absolute difference(normalize(z), normalize(goal))
```

The solved-state encoding is used only to construct training labels. It is not
an input to the deployed planner.

## Primary loss

All valid rollout endpoints in the training example are compared with a
logistic pairwise ranking loss. If true endpoint `i` is closer to the goal
than `j`, the learned Energy of `i` should be lower.

The primary endpoint-ranking weight is 1. There is no absolute endpoint-Energy
regression.

## Small auxiliary distillation

The validated configuration retains a 0.25-weight pair-difference regression
on one-step candidate Energies. Its targets are differences between the best
sampled continuation distances for each root. This is already a weak form of
distilling multistep quality onto the first action.

Because mixed horizons provide different targets for the same one-step form,
a dedicated fixed-H4 distillation head is a clean one-step improvement
experiment.

## Relation to GAR

| Method | Input being scored | Target |
|---|---|---|
| Local GAR | one predicted transition | local or distilled action order |
| Endpoint Energy | recursively predicted endpoint | quality of that endpoint |

The paper method uses Endpoint Energy. Local GAR appears only in ablations.
