# Planning

## Receding-horizon algorithm

At each real environment step:

1. Encode the observed history into `z_t`.
2. Enumerate root actions under the selected interface.
3. Build fixed-depth action beams.
4. Recursively predict every beam endpoint.
5. score each final endpoint once;
6. choose the lowest-Energy beam;
7. execute only its first action;
8. observe the true outcome and replan.

No edge Energies are summed.

## Root-balanced beam

Every first action retains up to `B=8` continuations at each depth. This
prevents an initially weak root from disappearing before its later
consequences can be evaluated.

## Beam score

```text
score(a_1,...,a_D)
  = EndpointEnergy(z_t, F^D(z_t,a_1,...,a_D), z_0, D)
```

Lower is better. The first action of the best final beam is executed.

## Terminal safety

A sequence that solves early keeps an absorbing endpoint, but its nominal
requested depth remains `D`. The head cannot infer success from a shortened
horizon argument.

## Candidate interfaces

### Feasible-tree evaluation

Current and imagined actions are enumerated from the reference dependency
graph. D>1 is candidate-privileged.

### Full-catalogue evaluation

Every problem action is considered at every search node. No feasible-action
menu is supplied. Invalid and repeated actions are imagined like any other
action and return an unchanged-state observation when executed in the real
environment.

## Compute axis

Primary test depths are `1,4,8,16`. Beam-width and FLOP sweeps are required
for the final compute-scaling figure.
