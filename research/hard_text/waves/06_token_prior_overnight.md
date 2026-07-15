# Wave 06: state-conditioned token support and hierarchical planning

## Question

Can an internal state-conditioned primitive-token prior keep hierarchical
oracle-goal planning on executable text support, without an auxiliary language
model or symbolic proposals? Does jointly training that prior alter the shared
state representation enough to improve multilevel planning?

## Implementation

- Added an optional token prior `p(a_t | s_t)` to the causal token hierarchy.
- Compared a linear and nonlinear prior, joint versus detached-state training,
  loss weights, label smoothing, hierarchy depth/stride/bottleneck, phase
  augmentation, and densely supervised rollout depth.
- Added four no-LM primitive planning controls:
  1. uniform categorical CEM;
  2. prior NLL as an energy inside categorical CEM;
  3. autoregressive prior shooting followed by oracle-goal selection;
  4. greedy prior rollout without goal-directed primitive search.
- Macro planning remains top-down conditional-bank CEM. The exploratory budget
  is 512 candidates and 8 CEM updates, within the lower end of the HWM paper's
  hundreds/thousands of candidates and 5--40 updates. Reachability refinement
  is crossed with the best prior-shooting proposal.
- Diagnostics include token-prior NLL/top-1/top-5/rank/ECE by token class and
  phase, multistep drift at every hierarchy level, and grouped symbolic linear
  probes of graph, state, feasibility, value, and next-step information.

The exact 21-cell matrix is in
`research/hard_text/overnight_token_prior_matrix.tsv`. Screens use 6,000 fresh
training examples, three epochs, three planning episodes, and at most 64
executed tokens. These are exploratory signals, not paper-level estimates.

## Deployment (2026-07-15)

| Site | Work | Scheduler identifier | Capacity |
|---|---|---|---|
| Grünau | seed 0, first 18 cells plus one remaining bottleneck cell; matched planning evaluations | direct SSH workers | 18 concurrently verified-free GPUs across servers 7, 10, 11, 12 |
| Alex | full 21-cell seed-2 array | `3861524`; diagnostics `3861541` | up to 8 RTX Pro 6000 GPUs |
| Lise | full 21-cell seed-1 array | retry `9334852`; diagnostics `9334868` | up to 4 shared A100 GPUs |
| Grete | remaining bottleneck/L3 cells | retry `14877185`; diagnostics `14877186` | up to 3 shared A100 GPUs |

The first Lise and Grete attempts exposed a missing optional TensorBoard
dependency and were cleanly retried after CSV-only logging fallback was added.
Several Grünau training jobs completed before a live update of their shared
worker script disrupted the evaluation loop; checkpoints were unaffected and
missing evaluations were relaunched with the immutable
`run_token_prior_eval_cell_v1.sh` worker. Do not modify that file while this
wave is active.

## Success criteria

The prior is useful only if it improves reference-token rank/calibration and
valid-sentence or task success under oracle-goal planning, rather than merely
lowering teacher-forced CE. A code-support diagnosis is supported if prior
shooting or prior-energy CEM beats uniform CEM, especially with lower planning
drift. A representation effect requires corresponding changes in symbolic
probe information or hierarchy-level drift, and must be distinguished from a
pure proposal-distribution effect using the detached-prior cells.

## First fast signal (seed 0)

The initial 18-cell screen and diagnostics completed before the exposure-aware
follow-up was launched.

- No initial cell produced a valid sentence or solved an episode. This rules
  out the teacher-forced state-conditioned prior, by itself, as a solution.
- Stronger joint token-prior supervision improved teacher-forced top-1 from
  roughly 0.59 (weight 0.1) to 0.66 (weight 5), but increased token-predictor
  four-step latent drift from roughly 0.05 to 0.44. The prior was changing the
  representation in a way that harmed open-loop dynamics.
- Detaching prior inputs preserved very low four-step drift (about 0.04) but
  capped top-1 near 0.58. This identifies a real prediction/support tradeoff,
  rather than a planner-only hyperparameter issue.
- Greedy and prior-shooting rollouts had substantially less latent drift than
  uniform categorical CEM and approached the oracle goal more closely, but
  still emitted no executable sentence. Latent goal proximity is therefore
  insufficient evidence of primitive text support.
- Feasible-action linear probes remained weak (about 0.55--0.58 balanced
  accuracy) and did not improve consistently with hierarchy or prior strength.

This motivated a second 13-cell wave that trains the prior on the low
predictor's one- through eight-step open-loop states. It crosses rollout-loss
weight, detached versus joint gradients, horizon, discount, nonlinear
capacity, and smoothing, with five seed-1 confirmations. Its first partial
signal contains isolated valid sentences but no solved episode; complete
matched evaluation and diagnostics are still running.

The L3 cell also exposed and now regression-tests a genuine implementation
bug: nested phase augmentation restarted each level's offsets at token zero and
could request nonexistent lower macro actions. Higher levels now consume the
actual valid lower-level grid and inherit its absolute phase offset.
