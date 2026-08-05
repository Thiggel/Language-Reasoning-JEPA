# Training targets

## Sampling

For each generated problem:

1. Generate one valid solution trajectory.
2. Sample one anchor position.
3. Sample training horizon `H` from `{1,2,4,8}`.
4. Form three root actions: factual plus two feasible alternatives.
5. Sample four environment continuations per root.
6. Stop a continuation when the query is solved or no action is feasible.
7. Pad the remaining requested horizon with absorbing no-ops.

The requested `H`, not the number of executed actions, is supplied to the
Energy head. This prevents early solution length from leaking the label.

## Targets and predictions

For each sampled sequence:

```text
teacher endpoint = EMA encoding of the environment-generated final history
model endpoint   = online predictor applied H times to the action sequence
label            = normalized latent distance(teacher endpoint, EMA goal)
Energy           = Head(anchor, model endpoint, initial state, H)
```

The loss ranks every valid rollout endpoint, not only the best rollout beneath
each root.

## Counterfactuals

Alternative root actions receive true one-step outcome targets and latent
prediction loss. Their longer continuations also contribute endpoint-ranking
examples. Thus the method uses counterfactual training data.

Required control:

```text
same factual and counterfactual transitions, endpoint-ranking weight = 0
```

## Current privilege

Training continuations after each root are sampled from the symbolic feasible
menu. This is environment supervision during training. The new full-catalogue
variant additionally samples invalid root actions and represents them with an
unchanged-state outcome.

## Cost

The winning recipe uses no dense loss at intermediate imagined states.
Backpropagation still traverses every predictor application because the final
endpoint Energy depends on the complete recursive rollout.
