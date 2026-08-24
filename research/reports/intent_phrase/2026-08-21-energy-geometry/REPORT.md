# Energy geometry: the goal ruler was measured in the wrong space

2026-08-21. Flat-backbone intent JEPA, faithful iGSM.

## 0. The short version

The standing story was that the latent goal-distance ruler is *blunt* — it
reaches the goal but wanders, and wanders worse with depth — and that this was
evidence about the learned representation. It is not. It is a **metric bug in
the diagnostic scorer**.

`latent_pred` trains the predictor against **LayerNorm-ed** targets
(`objectives/base.latent_distance` LayerNorms *both* sides), so the predictor
is only ever asked to match the encoder **up to LayerNorm**. Raw scale and
offset are unconstrained, and in the trained model they diverge: encoder states
have mean L2 norm **66.6**, predicted states **147.8** — a **2.22x** gap. But
`flat_search._score` scored `oracle_distance` with **raw L2** against an
encoded goal. The scorer was comparing two different spaces.

Re-measuring the same quantities in the space the predictor is actually trained
in (LN-L1) flips every conclusion that depended on it.

## 1. Measurement

`scripts/measure_energy_monotonicity.py` (new). Walks the ground-truth
solution, encodes every prefix, and at every true state ranks every feasible
action under both rulers. No planner, no generation, no step cap. 200
validation problems, `_ckpt_snapshots/ecf16-s0-best-2026-08-20.pt`.

**ORACLE DIAGNOSTIC**: the goal is built from the ground-truth solved solution
and the feasible sets come from the environment. Measurement only; no row here
is a deployable policy or a planning result.

| statistic | raw L2 (what the scorer used) | LN-L1 (the trained-in space) |
|---|---|---|
| true next action ranked top1 by distance | **.109** | **.822** |
| distance-argmin is a necessary action | **.185** | **.898** |
| predicted endpoint closer to goal than current state | **.000** | **.896** |
| descend_frac along the true trajectory | .793 | .789 |
| descend_frac excluding the trivial final step | .748 | .742 |
| Kendall tau of goal distance vs step index | -.799 | -.790 |

Learned energy head, same protocol (metric-independent — it consumes raw
vectors): true-action top1 **.872**, percentile **.077**, argmin-is-necessary
**.951**.

Encoder state norm **66.6**; predicted state norm **147.8**; ratio **2.22**.

## 2. What this establishes

1. **Monotonicity is not the problem.** The goal potential decreases at ~79% of
   steps with tau ~-.79 in *both* metrics. The hypothesis "the potential is not
   monotone, so descending it is not even trying to follow the true
   trajectory" is rejected. (It is not perfectly monotone either — ~21% of
   steps move away — so a monotonicity term may still buy something, but it is
   not the main lever.)
2. **Under the correct metric the goal geometry is good**: the distance ruler
   picks a necessary action 90% of the time, and predicted endpoints move
   toward the goal 90% of the time. Under the raw metric a predicted endpoint
   is *never* closer to the goal than the state already occupied — which is
   precisely the pathology that makes descent wander.
3. **The energy head was never affected.** `HorizonEnergyHead` begins with a
   `LayerNorm` over its concatenated input, so it absorbs the scale offset.
   That is why energy ranks well (top1 .872) where raw distance does not
   (.109).
4. **It explains `endpoints=true` being the worst arm.** `union-d2-trueend`
   scored .465 env / .285 answer, below the random-policy bound. Feeding
   *encoder* states (norm ~67) to an energy head trained on *predicted* ones
   (norm ~148) is a 2.2x off-distribution shift in its candidate slot.

## 3. Consequence for the record

Every `oracle_distance` row in `2026-08-20-true-oracle-upper-bound` is
contaminated and must be re-run in LN-L1 before being cited. The
`symbolic_oracle` and `energy` rows are unaffected.

Round `2026-08-21-lnl1-ladder-v1` re-ran the ladder at depths 1/2/4/8 in
LN-L1, on that round's own checkpoint
(`2026-08-20-aggregate-ablation-v1/ckpt.pt`), same protocol (full_catalogue,
max-expand 64, 200 episodes). Result:

| depth | raw env / exact | LN-L1 env / exact | LN-L1 steps |
|---|---|---|---|
| 1 | .955 / .183 | **1.000 / .610** | 7.4 |
| 2 | .915 / .164 | **.985 / .371** | 8.4 |
| 4 | .920 / .038 | **.990 / .303** | 8.7 |
| 8 | .890 / .011 | **.980 / .352** | 9.0 |

**There is no depth collapse.** Raw exact-necessary fell 16x monotonically from
d1 to d8; in LN-L1 it drops once (d1 -> d2) and then flattens, with d8 (.352)
ABOVE d4 (.303). Env success is .980-1.000 at every depth and mean steps grow
only 7.4 -> 9.0. The entire "collapses with depth" phenomenon was the scale
mismatch compounding through recursive predictor applications. The remaining
question is the single d1 -> d2 step, which is a much smaller and better-posed
problem.

**Caveat on one number**: the raw-metric `argmin_is_necessary` of .185 and the
planner's depth-1 `exact_necessary` of .183 agree closely, but they come from
**different checkpoints** (md5s differ). Suggestive, not a reproduction.

## 4. Two ways to fix it, and which to pick

- **Score in LN-L1** (`plan_flat --distance-metric ln_l1`). Free, no
  retraining. This is not a patch over a defect — LN-L1 *is* the equivalence
  class the predictor was optimized in, so it is the honest metric for any
  comparison that mixes predicted and encoded states. Use it for every
  existing diagnostic immediately.
- **Train the predictor in raw space**
  (`objective.latent_pred.norm_targets=false`, likewise
  `counterfactual_state`). The better long-run answer *if it trains stably*:
  then encoder, predictor, energy head and planner all share one geometry and
  no consumer has to remember which space it is in. The risk is real —
  normalized targets are a standard anti-collapse device in JEPA training, and
  VICReg is the only other thing holding that line.

These are not exclusive and both are in flight. The sweep decides whether the
raw-target variant is stable enough to become the recipe.

## 5. Components added to the flat path

None of the geometry losses existed on the flat training path; they lived only
in the old two-hierarchy `DiscourseJEPA`. A code survey also confirmed a real
structural gap: **no loss anywhere in the flat path compares an energy at
timestep t against t', or depth h against h'.** Every softplus pair shares an
identical root, initial state and horizon. Nothing puts different depths on a
common ruler.

Added (all default weight 0.0, no behaviour change when off):

| key | what it does | new params? |
|---|---|---|
| `objective.straighten` | `TemporalStraightening`: align consecutive latent velocities so Euclidean distance approximates minimum-step distance | no |
| `objective.monotone` | `GoalMonotonicity`, **label_free** — every observed step must reduce LN-L1 distance to the terminal EMA state; reads no necessary/distractor annotation | no |
| `objective.energy_monotone` | **new** `EnergyMonotonicity`: from one fixed root, `E(s_0, s_t, s_0, t)` must fall as t grows — the missing cross-time constraint | no |
| `objective.hindsight_monotone` | **new** `HindsightGoalMonotonicity`: the same constraint with the goal relabeled to a *random observed future state*, O(T^2) constraints instead of O(T) | no |
| `model.energy_head_kind=quasimetric` | **new** `QuasimetricEnergyHead`: energy as a Metric-Residual-Network distance to a predicted goal; non-negativity and the triangle inequality hold by construction | **yes** |

Also fixed: `GoalMonotonicity(label_free=True)` still dereferenced
`batch["necessary"]`, a key the flat pipeline never provides — it would have
crashed on first use.

### Why hindsight relabeling is worth having

The terminal state is one goal per trajectory, so terminal-only supervision
gives O(T) constraints and only ever describes "distance to done". Hindsight
relabeling treats any observed future `s_j` as a goal for the prefix before it:
the trajectory demonstrably reached `s_j`, so `s_t -> s_j` is a true
reachable-in-(j-t)-steps pair. Same data, O(T^2) constraints, and it shapes the
whole metric instead of one direction in it. Steps at or after the sampled goal
index are masked. Self-supervised — it reads only the observed order.

### Why a quasimetric head is worth having

The default energy is an unstructured MLP over
`concat[root, endpoint, initial, horizon]`. Nothing forces the energy at depth
h to compose coherently with depth h+1 — which is exactly the "not on a common
ruler" gap. The quasimetric head parameterizes

    E(endpoint, initial) = d_q(endpoint, G(initial))

with `G` a `GoalHead` and `d_q` a Metric Residual Network (Liu et al. 2023):

    d_q(x, y) = ||phi(x) - phi(y)||_2 + max_i relu(psi(y)_i - psi(x)_i)

symmetric metric plus asymmetric quasimetric. Non-negativity and the triangle
inequality hold **by construction**, so coherent descent is structural rather
than propped up by a penalty. `root` and `horizon` are ignored on purpose: the
point is a single state potential shared by every depth. The goal head is
trained implicitly by the ranking terms and, when enabled, by
`energy_monotone`.

Because it replaces the energy head it **adds parameters**, so warm starting
from an older checkpoint needs `init_allow_missing=true` (those parameters
start fresh; unexpected keys are still refused).

## 6. Normative note

Per the project owner (2026-08-21): it is acceptable for the energy to end up
encoding remaining steps, as long as the *supervision* is not symbolic —
everything must come from JEPA geometry. All five components above are
self-supervised: they read the observed trajectory order and the EMA-encoded
states, never a step counter, a necessary/distractor annotation, or any
symbolic label. The one place a step index enters is
`HorizonEnergyHead`'s horizon input, which predates this work and is `false`
(horizon-blind) in the frozen recipe.

## 7. Status

- `2026-08-21-lnl1-ladder-v1` — corrected oracle ladder, d1/d2/d4/d8, running.
- `2026-08-21-geometry-sweep-v1` — six training cells, each warm-started from
  the ecf16 `last.pt` and trained further, so **every cell is also a
  train-longer arm**: `longer` (control), `rawpred`, `straighten`, `monotone`,
  `energymono`, `rawpred-energymono`. Queued behind a GPU poller.
- Not yet queued: `quasimetric` and `hindsight` cells (both smoke-tested).

The ecf16 run this all builds on **crashed** at step ~22.1k of a planned 62.5k
(`RuntimeError: Too many open files`, `state=FAILED`), with its energy AUCs
still rising. "Train longer" was never actually tested; every sweep cell now
tests it as a side effect.
