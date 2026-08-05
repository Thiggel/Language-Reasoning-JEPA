# Planning and beam search

## Closed-loop planning

Planning is receding-horizon. The model never commits to an entire imagined
sequence in the real environment.

At each real step:

1. Encode the prompt and all observed outcome sentences into `z_t`.
2. Obtain the currently feasible intent phrases.
3. Imagine candidate action sequences.
4. Select the best sequence under the configured Energy rule.
5. Execute only its first action in the true environment.
6. Observe and encode the true outcome.
7. Replan from the corrected state.

This limits compounding model error because every real action is followed by a
fresh observation.

## Depth-1 selection

At D=1, every currently feasible action is encoded and its successor is
predicted. The lowest-Energy action is executed.

This is the main shared-menu protocol and does not require future feasible
menus.

## Genuine global beam search

For D>1, the evaluator now performs iterative beam search rather than random
shooting:

```text
beam_1 = all currently feasible one-action sequences

for depth k = 1 ... D:
    score every current partial sequence
    keep the globally best B_eval sequences
    expand each survivor with its next feasible actions

execute the first action of the best final sequence
```

Current evaluation width is `B_eval=8`.

This is a global test-time beam, unlike the root-balanced training teacher.
Global pruning is appropriate for choosing one plan, but it creates a possible
failure mode: a promising root can be removed because its intermediate score
is poor even if a later continuation would have won.

## Candidate privilege at D>1

Future action availability is computed from the symbolic dependency graph
under each hypothetical prefix. The predictor supplies consequences and the
Energy supplies scores, but the environment supplies the future candidate
tree.

Every D>1 result must therefore be labeled candidate-privileged. It tests
consequence simulation and valuation while holding proposal feasibility fixed.

## Predictor rollout

For a candidate sequence `(a_0, ..., a_(D-1))`:

```text
z_hat_1 = F(z_t, a_0)
z_hat_2 = F(z_hat_1, a_1)
...
z_hat_D = F(z_hat_(D-1), a_(D-1))
```

For a causal Transformer, the observed state-action prefix is retained and
imagined states/actions are appended autoregressively. For the MLP predictor,
each transition is applied to the preceding predicted state.

## Correct transition-Energy composition

The intended score of a beam is the Energy at its end:

```text
score_terminal
  = E(z_hat_(D-1), z_hat_D, z_0)
```

Lower is better. Earlier transition Energies are not added.

This is now the default `transition_energy_composition=terminal`.

## Composition controls

### Cumulative control

```text
score_cumulative
  = E(edge_1) + E(edge_2) + ... + E(edge_D)
```

This was used accidentally in the first corrected-beam round. It double-counts
overlapping multi-step targets and is retained as a negative control.

### Root-only control

```text
score_root = E(z_t, z_hat_1, z_0)
```

This ignores imagined later consequences. Its chosen root should reproduce the
depth-1 policy, apart from tie and pruning details. It tests whether additional
simulation contributes beyond the distilled one-step head.

### Terminal state Energy

```text
score_state = V(z_hat_D, z_0)
```

This is natural when `V` predicts absolute goal distance. It was the only
overnight configuration that improved modestly at depth four under the first
beam evaluation, which helped localize the cumulative-composition error.

## Absorbing solved beams

If a hypothetical sequence reaches the query early, later positions are
represented as absorbing no-ops. Candidate sequences keep a common nominal
depth, preventing shorter terminal paths from leaking that they solved the
problem.

## Choice of first action

After the final beam is selected, the planner executes its first action. If the
best final beam is:

```text
(a_3, a_8, a_2, a_5)
```

the real environment receives only `a_3`. The remaining actions are discarded
and will be reconsidered after observing the true consequence of `a_3`.

## Known planning failure modes

- **Rollout drift:** imagined states leave the distribution seen by the Energy
  head.
- **Intermediate pruning:** a good root is removed before its useful
  continuation appears.
- **Energy miscalibration:** scores are ordered locally but incomparable across
  imagined states.
- **Candidate privilege:** improvements rely on exact future feasibility.
- **Depth saturation:** D exceeds the number of meaningful actions available.
- **Tie or enumeration leakage:** stable candidate order determines equal-score
  decisions.

Each should have a dedicated control rather than being inferred from aggregate
success alone.
