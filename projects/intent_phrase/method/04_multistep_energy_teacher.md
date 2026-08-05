# Multi-step Energy teacher

## Direct answer

Yes. A teacher horizon H=N means:

```text
take candidate first action a_0;
choose an approximately best continuation a_1, ..., a_(N-1);
imagine the endpoint with the EMA JEPA predictor;
measure that endpoint against the EMA goal;
attach the result as the target for a_0.
```

The first action is fixed separately for every candidate. The teacher searches
only beneath that root.

## Why use a multi-step target?

A one-step action can look locally unhelpful while enabling a strong later
continuation. The intended H-step target asks a more useful question:

> If I take this action now and act well afterward for H−1 steps, how good is
> the resulting state?

This distills limited-horizon planning into a one-step Energy head. At test
time, depth-1 action selection can benefit from training-time search.

## Exact algorithm for H>1

For an anchor history and each root candidate `a_0`:

1. Start from the EMA-encoded anchor state.
2. Encode every problem action phrase once and cache its detached action code.
3. Predict the root successor with the EMA predictor.
4. Track which semantic actions would be resolved after the root.
5. Enumerate the next symbolically feasible actions.
6. Predict each child in latent space. Do not execute it to obtain a true
   outcome sentence.
7. Compare predicted leaves by normalized latent distance to the EMA terminal
   goal.
8. Keep the best `B_teacher` leaves for this root.
9. Repeat until H actions have been proposed or the symbolic query is solved.
10. Use the minimum final latent distance found beneath the root as its label.

Current teacher beam width is 4.

## Root-balanced search

Teacher beams are maintained independently beneath each first action. If there
are three roots and width four, each root may keep four continuations.

This is essential for training labels. A global beam could discard a weak root
immediately, leaving no endpoint with which to supervise it. GAR needs a label
for every candidate in the same state.

## In what sense is the continuation “optimal”?

It is approximately optimal under the teacher's current geometry:

```text
teacher objective = minimum EMA latent distance to EMA goal
```

It is not guaranteed to be optimal in the environment. The teacher does not
consult:

- exact remaining necessary steps;
- membership in the query's ancestor set;
- exact continuation advantage;
- exact environment outcome values for hypothetical later actions.

The teacher does use the symbolic graph to determine which actions are
feasible after a hypothetical prefix. Therefore it is candidate-privileged.
The precise phrase should be:

> root-balanced, geometry-greedy continuation under a symbolic feasible-action
> tree

rather than unqualified “oracle optimal policy.”

## H=1 is a special case

For H=1 there is no continuation policy. The current implementation obtains
the label from the EMA encoding of the true one-step environment outcome.

For H>1, later endpoints are imagined by the EMA JEPA predictor. This means H=1
and H>1 differ not only in horizon but also in whether the endpoint itself is a
true encoded outcome or an imagined one. That is a real design detail and a
possible ablation if the horizon result needs stricter attribution.

## Target attached to the first transition

Let `leaf_H(a_0)` be the best endpoint found beneath root `a_0`.

Distance target:

```text
target_distance(a_0) = d(leaf_H(a_0), goal)
```

Advantage target:

```text
target_advantage(a_0)
  = d(leaf_H(a_0), goal) - d(current_state, goal)
```

The transition Energy head still sees only the current state and the predicted
one-step successor of `a_0`. Training asks it to infer the longer-horizon
quality that follows from that first consequence.

## Teacher horizon is not planner depth

Suppose a checkpoint is trained with H=8 and evaluated with beam depth D=4:

- H=8 determined the target used to train the Energy attached to each first
  transition.
- D=4 determines how many actions the online predictor imagines at test time
  before comparing beams.

The H8 head can also be used with D=1. In fact, the August screen found its
strongest current result at D=1, showing that multi-step teacher search was
successfully distilled into one-step selection.

## Why multi-step Energies must not be summed

An H8 Energy already summarizes the quality of a root after an H−1-step good
continuation. If a D-step beam sums H8 Energies at every edge, future progress
is counted repeatedly with heavily overlapping horizons.

The deployed beam score is now terminal-only:

```text
beam = (a_0, ..., a_(D-1))

beam_energy
  = E(z_hat_(D-1), z_hat_D, z_0)
```

The earlier cumulative rule is retained only as a negative control.

## Computational behavior

The teacher avoids repeatedly encoding rendered outcome text:

- action phrases are encoded once per problem and cached;
- endpoints are generated with predictor calls;
- paths with compatible prefix lengths are batched;
- EMA modules are frozen and deterministic.

For the current MLP checkpoints, observed throughput remained sufficient for
the complete 300,000-example H16 run to finish in under two hours of training.

## Open validity questions

- Does H>1 improve because of longer consequence reasoning or because the
  target is generated from predicted rather than true endpoints?
- How often does teacher beam width four match exhaustive geometry search?
- How often does geometry-selected continuation agree with exact environment
  continuation quality?
- Does the H8 benefit survive five seeds and length-OOD evaluation?
- Does the causal Transformer predictor reproduce the MLP horizon result?
