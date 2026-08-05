# Geometric Advantage Ranking

## Name and central purpose

GAR means **Geometric Advantage Ranking**. It is a training constraint that
orders candidate actions using the geometry of their successor states. It is
not, by definition, a particular score-head architecture or search algorithm.

The central problem is that accurate latent prediction does not determine a
planning metric. A predictor can represent the transition system correctly
under many invertible rearrangements of latent space. Those rearrangements can
change which successor appears closest to the goal.

GAR supplies the missing local ordering:

```text
if action a_plus has a better continuation than action a_minus,
then Energy(h, a_plus) should be lower than Energy(h, a_minus).
```

## Candidate set at one anchor state

Training chooses an anchor step `t` from a generated trajectory. The candidate
set contains:

- the factual action taken by the trajectory;
- `K` alternative feasible actions;
- optionally grounded invalid alternatives in older full-catalogue controls.

For each candidate `a_i`, the model predicts a successor:

```text
z_hat_i = F_online(z_t, u(a_i))
```

The teacher produces a lower-is-better label `d_i`. For H=1, `d_i` is the
distance between the EMA encoding of the true one-step successor and the EMA
goal. For H>1, `d_i` is obtained by the multi-step teacher described in the
next document.

## Distance used by the teacher

The current geometric distance is mean absolute difference after separately
normalizing each latent across its feature dimension:

```text
normalize(z) = layer_norm(z)

d(z, g) = mean_j abs(normalize(z)[j] - normalize(g)[j])
```

The goal `g` is the EMA-encoded terminal state of the generated solution
trajectory. This is a training target and a diagnostic privilege. The deployed
transition Energy head does not directly receive this terminal goal vector.

## Pairwise ranking loss

For every valid ordered pair where candidate `i` is geometrically better than
candidate `j` by more than a small label gap:

```text
d_i + label_gap < d_j
```

GAR applies a margin loss:

```text
rank_loss(i, j) = max(0, margin + E_i - E_j)
```

The loss is zero only when the better action has Energy lower by at least the
configured margin.

Current Energy experiments use:

- ranking weight: 1.0;
- margin: 0.5;
- label gap: 0.02.

## Absolute calibration loss

Ranking determines order but not scale. Planning can require comparable
scores across imagined states and depths, so the current default also regresses
each Energy to an absolute target:

```text
energy_mse = mean_valid (E_i - target_i)^2
```

The default regression weight is 0.25. Ablations use 0, 0.1, 0.5, and 1.0.

Two targets exist:

### Distance target

```text
target_i = d(endpoint_i, goal)
```

This is an absolute endpoint Energy.

### Advantage target

```text
target_i = d(endpoint_i, goal) - d(current_state, goal)
```

Negative values indicate predicted progress. Positive values indicate moving
away from the goal. The word “advantage” here means a geometric progress
surrogate, not necessarily a Bellman action advantage.

## Gradient routes

In the default predicted-state variant:

```text
GAR loss
  -> transition Energy head
  -> predicted successor
  -> online JEPA predictor
  -> online state and action representations
```

The teacher label is stop-gradient. Thus GAR can shape the predictive geometry
without changing the target to make the loss easier.

In the true-state ablation, the Energy head receives EMA true states. GAR then
trains the head but does not send its gradient into the online predictor through
the Energy input. This separates “a useful supervised head” from “GAR shapes
imagined geometry.”

## GAR is not the direct ranker

Both use pairwise supervision, but their factorization differs:

```text
GAR transition scorer: z_t, a -> predicted successor -> Energy
direct ranker:          z_t, a -> score
```

The direct ranker is a necessary control. If it matches GAR everywhere, a
reviewer can argue that successor prediction is unnecessary. Evidence for the
predictive factorization should come from transfer, paraphrase robustness,
data efficiency, or improved behavior under simulation.

## Counterfactual data versus ranking

GAR experiments also train counterfactual latent prediction. Therefore the
method must be compared against a control that receives the same alternative
transitions but no ordering loss. Otherwise an improvement could come merely
from seeing more counterfactual outcomes.

The clean mechanism comparison is:

- factual latent prediction only;
- factual plus counterfactual latent prediction;
- counterfactual prediction plus absolute Energy regression;
- counterfactual prediction plus GAR ranking;
- full ranking plus calibrated regression.

## What GAR claims and does not claim

GAR directly claims local action ordering within a state. It does not by itself
guarantee:

- a globally Euclidean state space;
- a Bellman-consistent value function;
- stable long-horizon rollout;
- calibrated comparison after arbitrary repeated composition;
- correct proposal of actions outside the supplied candidate interface.
