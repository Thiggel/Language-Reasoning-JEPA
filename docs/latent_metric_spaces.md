# Latent metric spaces, and the geometry knobs on the flat path

Read with `docs/flat_intent_jepa_design.md`. This file exists because a
mismatch between two latent spaces silently corrupted a whole family of
diagnostics; the rule at the top is the part worth remembering.

## The rule

> **Predicted states and encoded states are only comparable up to LayerNorm.
> Any distance that mixes them must be measured in LN space.**

`objective.latent_pred` (and `counterfactual_state`) are computed by
`objectives/base.latent_distance`, which — with the default
`norm_targets: true` — applies `F.layer_norm` to **both** the prediction and
the target before comparing. The predictor is therefore never asked to
reproduce the encoder's raw scale or offset, only its LayerNorm-ed direction.

Measured on `ecf16-s0-best-2026-08-20` (200 validation problems):

| quantity | value |
|---|---|
| mean L2 norm of encoder states | 66.6 |
| mean L2 norm of predicted states | 147.8 |
| ratio | **2.22x** |

Consequences, all confirmed by measurement
(`research/reports/intent_phrase/2026-08-21-energy-geometry/REPORT.md`):

- A **raw L2** distance from a predicted endpoint to an encoded goal is
  dominated by the scale offset. Under raw L2 a predicted next state is
  *never* closer to the goal than the state already occupied (0/200 problems);
  under LN-L1 it is closer 90% of the time.
- Ranking feasible actions by raw-L2 distance to the encoded goal picks a
  necessary action **18.5%** of the time. The same ranking in LN-L1 picks one
  **89.8%** of the time.
- The learned energy head is immune, because `HorizonEnergyHead` starts with a
  `LayerNorm` over its concatenated input. This is why energy ranked well
  (top-1 .872) while raw distance did not (.109) — the difference was the
  metric, not the quality of the representation.

### Practical guidance

- Scoring / diagnostics: use `plan_flat --distance-metric ln_l1`. `raw` is the
  default only to keep old runs reproducible; **it is the wrong metric for any
  comparison that mixes predicted and encoded states**. `flat_search.goal_distance()`
  is the single place both are implemented.
- Feeding **encoder** states to a head trained on **predicted** ones (e.g.
  `plan_flat --endpoints true`) is a 2.2x off-distribution shift in the
  candidate slot. Expect it to underperform the "privileged" label it carries.
- If you want one geometry everywhere, train with
  `objective.latent_pred.norm_targets=false` (and the same on
  `counterfactual_state`). Caveat: normalized targets are a standard
  anti-collapse device in JEPA training; with them off, VICReg is the only
  remaining guard. Verify representation health, not just planning success.

## Where energy comparisons happen (and what is missing)

Every energy term on the flat path compares candidates that share an
**identical root, initial state and horizon**:

- `flat_intent_jepa._geo_rank` (anchor contrast) — candidates one predictor
  step from the same real anchor.
- the depth contrast along imagined prefixes — executed vs infeasible intents,
  both one step from the *same* imagined prefix.

Both populations are concatenated into the same tensors, so the loss spans
several imagined depths, but **no pair ever compares an energy at timestep `t`
against `t'`, or depth `h` against `h'`.** Nothing puts different depths on a
common ruler. `objective.energy_monotone` is the term that closes this gap.

## Geometry knobs

All default to `weight: 0.0` / `mlp`, so the frozen recipe is unchanged when
they are off.

| key | effect | adds params |
|---|---|---|
| `objective.straighten.weight` | `TemporalStraightening` — align consecutive latent velocities so Euclidean distance approximates minimum-step distance | no |
| `objective.monotone.{weight,margin}` | `GoalMonotonicity` (always `label_free` here) — every observed step must reduce LN-L1 distance to the terminal EMA state | no |
| `objective.energy_monotone.{weight,margin}` | `EnergyMonotonicity` — from one fixed root, `E(s_0, s_t, s_0, t)` must fall as `t` grows | no |
| `objective.hindsight_monotone.{weight,margin,n_goals}` | `HindsightGoalMonotonicity` — same constraint, goal relabeled to a random observed future state | no |
| `model.energy_head_kind` | `mlp` (default) or `quasimetric` — energy as a Metric-Residual-Network distance to a predicted goal | **yes** |
| `init_allow_missing` | permit warm start when new flags add modules; those parameters start fresh. Unexpected keys still fail. | — |

### `energy_monotone`

From the single fixed root `s_0`, require `E(s_0, s_t, s_0, t)` to decrease in
`t`, hinged with `margin`. This is the cross-time calibration described above:
it is the only term that ties energies at different depths to one scale.
Self-supervised — it reads the observed trajectory order and nothing else.

### `hindsight_monotone`

The terminal state is one goal per trajectory, so terminal-only supervision
gives O(T) constraints and only ever expresses "distance to done". Hindsight
relabeling treats **any** observed future `s_j` as a goal for the prefix before
it: the trajectory demonstrably reached `s_j`, so `s_t -> s_j` is a true
reachable-in-`(j-t)`-steps pair requiring no extra data and no symbolic label.
Same trajectories, O(T^2) constraints, and it shapes the whole metric rather
than one direction in it. Steps at or after the sampled goal index are masked.
`n_goals` independent goals are sampled per batch; the term is stochastic and
can be a near no-op on a very small batch (verified to produce encoder
gradients in 20/20 draws at realistic shapes).

### `energy_head_kind: quasimetric`

The default energy is an unstructured MLP over
`concat[root, endpoint, initial, horizon]`; nothing forces depth `h` to compose
coherently with depth `h+1`. `QuasimetricEnergyHead` parameterizes instead

    E(endpoint, initial) = d_q(endpoint, G(initial))

with `G` a `GoalHead` and `d_q` a Metric Residual Network:

    d_q(x, y) = ||phi(x) - phi(y)||_2 + max_i relu(psi(y)_i - psi(x)_i)

a symmetric metric plus an asymmetric quasimetric. Non-negativity and the
triangle inequality hold **by construction**, so coherent descent is structural
rather than propped up by a penalty. `root` and `horizon` are accepted for
signature compatibility and deliberately ignored — the point is one state
potential shared by every depth. The goal head is trained implicitly by the
ranking terms and, when enabled, by `energy_monotone`; it has no separate
regression target.

Because it replaces the energy head it adds parameters: warm starting from an
older checkpoint requires `init_allow_missing=true`.

## Self-supervision status

All of the above read only the observed trajectory order and EMA-encoded
states. None reads a step counter, a necessary/distractor annotation, or any
symbolic label, so all are admissible under the no-symbolic-heads rule in
`CLAUDE.md`. Per the project owner (2026-08-21) it is acceptable for the energy
to *end up* encoding remaining steps; what is forbidden is supervising it that
way.

The one pre-existing step-index input is `HorizonEnergyHead`'s `horizon`
argument, which is `geo_horizon_input: false` (horizon-blind) in the frozen
recipe.

## Measurement tool

`scripts/measure_energy_monotonicity.py` walks ground-truth solutions and
reports, in **both** metrics: descent fraction and Kendall tau of the goal
potential, per-step margin in units of the candidate spread, top-1 /
percentile rank of the true next action under each ruler, whether the
distance-argmin is a necessary action, predicted-vs-encoded endpoint distance,
and the encoder/predictor norm ratio. No planner, no generation. It is an
oracle diagnostic (goal and feasible sets come from the environment) and must
never be quoted as a planning result.
