# Cycle: token-level prerequisite support

Status: implementation validated; bounded mechanism pilot planned

## Decision

Determine whether preserving lexical tokens between candidate intents and the
causal executed-action prefix fixes the remaining deployable availability
errors better than phrase-level pooling or equal-capacity masked history.

## Falsifiable comparison

Initialize every model from the same aligned phrase-history checkpoint. The
primary head encodes candidate tokens and applies multi-head token attention to
all earlier executed or imagined intent tokens. A negative control uses the
identical head and parameters but masks history. Frozen JEPA encoders,
predictor, value head, action catalogue, behavioral prior, feasibility labels,
training examples, dropout zero, and evaluation protocol remain matched.

Train aligned token heads at learning rates 1e-3 and 3e-3 and the masked token
control at 3e-3 for five epochs on 40,000 fresh problems. Re-evaluate both
existing phrase-history learning-rate checkpoints on the same 120 episodes.
Tune each head over support weights 1, 3, 10, and 30 rather than assuming its
logit scale matches the pooled head.

## Validity and information boundary

The token head sees only intent phrases in the prompt catalogue and actions
already executed or imagined inside its own beam. It never receives variable
identifiers, graph edges, resolved sets, symbolic feasibility, outcomes from
unexecuted actions, or future feasible menus. Feasibility labels are explicit
training supervision and must remain disclosed. Candidate order is stably
shuffled. Exact-length audits use fixed length-six and length-nine problems.

Tests require strict prefix masks, empty-prefix finiteness, masked-control
history invariance, gradients confined to the support/prior heads, padding
safety, and non-oracle depth-two execution. Any missing artifact, non-finite
loss, checkpoint mismatch, or candidate-information violation invalidates its
cell rather than being silently excluded.

## Primary metrics and result patterns

Primary metrics are length-nine prior-only plus-two success and invalid-action
rate. Health metrics are true-state, one-step-predicted, and open-loop support
accuracy, positive/negative recall, and feasible-versus-infeasible pair
accuracy. Length-six strict success checks whether a gain is limited to the
harder regime.

- If aligned token history reaches invalid rate below .25, beats both pooled
  checkpoints and the masked token control, and improves plus-two success, keep
  it and confirm with independent training seeds.
- If aligned and masked token heads improve equally, extra token capacity—not
  prerequisite history—explains the gain; simplify to the masked architecture.
- If token-level pair accuracy improves but closed-loop validity does not, the
  behavioral prior/candidate composition is the next bottleneck.
- If neither aligned token learning rate beats pooled history, stop custom
  availability heads and move to a token policy language model.
