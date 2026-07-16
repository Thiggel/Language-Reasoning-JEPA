# Wave 08 - faithful hierarchical planning

Status: historical mechanism-development wave. Superseded by Wave 10: dense
recursive supervision improved the model, but a new 3×500 paired confirmation
found .289 hierarchy versus .324 flat. No hierarchy is currently selected.

## Scientific question

Can a compressed intent-action span predict a reachable long-horizon latent
subgoal, and can a low-level feasible-action planner use that subgoal without
access to oracle future actions?

## Screening matrix

| Axis | Initial cells | Held fixed |
|---|---|---|
| Macro dimension | 2, 4, 8, 16, 32 | K=3, deterministic |
| Macro stride | 2, 3, 4, 6 | selected dimension |
| Macro distribution | deterministic; Gaussian q plus conditional p | selected K/dim |
| High horizon | 1, 2, 4 macro steps | selected model |
| Planner | prior shooting; prior-CEM; retrieved macros | high-level value |
| Energy | oracle goal distance; distilled `V_hi` | same candidate samples |
| Support cost | 0, 0.1, 0.5, 1.0 times negative log prior | probabilistic model |

Reported prior-CEM uses the HWM Appendix-B regime: hundreds to thousands of
candidates, 15--40 high-level refits, fixed elite counts, and EMA-smoothed
variance. Reduced budgets are implementation smokes only.

## Required measurements

- direct high-level versus recursively composed low-level prediction error;
- high-level action sensitivity and macro-code effective rank;
- prior calibration and posterior/prior coverage;
- first-subgoal low-level reachability;
- strict/+2 end-to-end planning;
- success versus predictor evaluations and wall time;
- oracle-distance versus distilled-value gap.

Only the cells in this document may be launched. The hard-text hierarchy wave
is paused until this smaller environment yields a working hierarchical
controller.

## Active launch - 2026-07-14 09:47 CEST

All cells use stride `K=3`, a frozen initialized low-level model, a learned
high-level value, and only hierarchy losses. Each cell evaluates prior
shooting (1024 samples) and faithful HWM-style CEM (1200 samples, 20 refits,
10 elites, variance EMA 0.9), both at strict and +2 budgets on 100 episodes.

| Run | Macro dim | Macro model | GPU |
|---|---:|---|---|
| `intent_hier_k3_d2_det` | 2 | deterministic + conditional prior | gruenau12:0 |
| `intent_hier_k3_d2_var` | 2 | Gaussian posterior/prior | gruenau12:2 |
| `intent_hier_k3_d4_det` | 4 | deterministic + conditional prior | gruenau12:3 |
| `intent_hier_k3_d4_var` | 4 | Gaussian posterior/prior | gruenau12:4 |
| `intent_hier_k3_d8_det` | 8 | deterministic + conditional prior | gruenau12:5 |
| `intent_hier_k3_d8_var` | 8 | Gaussian posterior/prior | gruenau12:6 |
| `intent_hier_k3_d16_det` | 16 | deterministic + conditional prior | gruenau12:7 |
| `intent_hier_k3_d16_var` | 16 | Gaussian posterior/prior | gruenau12:8 |
| `intent_hier_k3_d32_det` | 32 | deterministic + conditional prior | gruenau12:9 |
| `intent_hier_k3_d32_var` | 32 | Gaussian posterior/prior | gruenau11:0 |

## Initial screen result

All ten cells completed. Strict / +2 success for HWM-scale CEM was:

| Macro dimension | Deterministic | Variational |
|---:|---:|---:|
| 2 | .05 / .32 | .11 / .46 |
| 4 | .20 / .56 | .24 / .52 |
| 8 | .11 / .46 | .21 / .61 |
| 16 | .34 / .52 | .13 / .45 |
| 32 | **.37 / .67** | .09 / .32 |

The deterministic 32-dimensional bottleneck is the screen winner, but .37
strict is not an adequate hierarchy. Variational macro actions are not a
reliable off-manifold solution and are paused.

## Failure localization and controls

The following controls use the same 100 validation episodes (random=.08,
flat learned-value planner=.84 strict).

| High subgoal | Lower controller | Strict | Interpretation |
|---|---|---:|---|
| True next K-step state | Oracle-feasible H=3 search | .84 | hierarchy interface can work |
| True next K-step state | One-step feasible selector | .20 | a K-step subgoal needs K-step control |
| True next K-step state | all problem actions, H=3 | .03 | future action availability is not learned |
| True next K-step state | continuous low CEM, 1200x20 | .08 | low CEM exploits action embeddings |
| Learned macro, discrete all-action spans | learned subgoal tracking | .15 | deployed macro waypoint is inaccurate |
| Learned macro, continuous high CEM | oracle-feasible lower search | .07 | high-level support/dynamics fail independently |

Twenty low-level CEM updates reduce the optimized objective substantially but
do not change behavioral success (.08), so inadequate CEM convergence is not
the explanation. The learned all-state action-availability head reaches .848,
.841, and .857 accuracy on true, one-step-predicted, and open-loop-predicted
states, respectively; it raises unrestricted subgoal tracking only to .21-.22.

High-level CEM also leaves the macro manifold. Without / with a soft support
penalty, the chosen code remains 3.07 / 2.98 RMS units from the nearest valid
macro code and the generated waypoint remains .746 / .734 normalized L1 from
the nearest true reachable state. A hard threshold does not improve this.

## Counterfactual macro repair

All cells use valid alternative three-action spans from the same state and
exact remaining necessary steps after each span.

| d_macro=32 cell | Macro dynamics L1 | V(s') pair acc. | Q(s,m) pair acc. | Support pair acc. |
|---|---:|---:|---:|---:|
| Base | .794 | .409 | .479 | .485 |
| + counterfactual dynamics | .529 | .419 | .462 | .515 |
| + state value | .529 | .822 | .461 | .515 |
| + action value | .529 | .419 | .799 | .515 |
| + action value and ranking | .529 | .419 | .831 | .515 |
| + state value, Q, ranking | .529 | .822 | .831 | .515 |
| + multi-negative support | .529 | .822 | .831 | 1.000 |

These heads learn their supervised tasks, but accurate ordering alone does
not create a reachable continuous macro plan. Direct execution of enumerated
spans reaches .80-.83 only when a large one-step value tie-break (weight
10-30) is added, which merely converges back to the flat .84 policy rather
than demonstrating a hierarchy gain.

## Reachability sweep result

The remaining diagnosed inconsistency is that the high predictor is trained
toward the encoded true K-step state while the lower planner operates through
the frozen low predictor. Its K-step imagined endpoint is .580 L1 from the
encoded target, and the repaired high prediction is .386 from that imagined
endpoint. `HierarchyReachability` therefore aligns each factual and
counterfactual high prediction with the endpoint obtained by rolling the
same low action span through the low predictor, while retaining the true-state
target.

| Reach weight | High-to-low L1 | High-to-true L1 | Valid macro + oracle lower | Discrete deployable | CEM + oracle lower | CEM deployable |
|---:|---:|---:|---:|---:|---:|---:|
| .25 | .347 | .509 | .07 | .15 | .08 | .09 |
| 1 | .294 | .520 | .07 | .14 | .03 | .09 |
| 4 | .242 | .548 | .06 | .11 | .03 | .09 |
| 16 | .228 | .568 | .06 | .09 | .04 | .12 |

Increasing reachability weight monotonically moves the high waypoint toward
the frozen low rollout, but simultaneously moves it away from the encoded
true state; its best behavioral change is only .09 to .12 in the noisy fully
deployed cell, while the oracle-lower control remains .03-.08. High-only
consistency is therefore rejected.

Multi-scale support improves local discrimination: support pair accuracy at
.25 empirical standard deviations rises from about .83 to .924, and the CEM
code's RMS distance to the nearest valid span falls from 2.98 to 2.21. It
still gives only .07 with oracle lower actions and .08 fully deployed. Exact
KNN-to-span penalties and projection also give only .05-.07. The failure is
therefore no longer attributable only to off-manifold codes.

Every completed checkpoint is evaluated in four stages: on-manifold discrete
macro plus oracle lower control; discrete all-action macro plus learned lower
support; continuous 1200x20 macro CEM plus oracle lower control; and the fully
learned continuous hierarchy. Selection is based on strict end-to-end
success, not the auxiliary loss.

## Active lower-dynamics and selector repair

The current sweep reverses the consistency direction. It unfreezes only the
low predictor, retains one-step latent supervision, and trains every factual
and counterfactual three-step low rollout toward the encoded true endpoint.
The high model remains trained on that same true endpoint. Cells use lower
rollout weights .25/1/4 without high-to-low consistency, plus one combined
weight-1/weight-1 cell.

The endpoint-only runs are now complete. They reduce recursive lower-model
K-step error from .580 to .483-.485, and the combined cell raises exact
recovery of a high waypoint's generating action span from .550 to .815
(same-first-action .575 to .815; mean retrieval rank 2.16 to 1.40). This is a
real interface improvement, but strict success remains .03-.04 for valid
macro/oracle-lower planning and .11-.15 for deployable span enumeration.

The retrieval audit identifies the remaining selector error: the Q-selected
macro starts with a query-relevant action only .56 of the time, even though Q
selects a terminal-optimal span on .915 of anchors. Terminal cost-to-go is
invariant to ordering within a span, while receding-horizon control executes
only its first action. The active `MacroRecedingValue/Ranking` objective keeps
terminal progress primary and uses discounted remaining distance after each
prefix to break terminal ties in favor of earlier progress.

In parallel, a selector diagnostic adds a listwise top-1 macro objective with
8 or 16 same-state counterfactual spans. The existing pairwise Q accuracy
(.831) hides a stronger top-1 result (.915 optimal, .125 mean regret) on
oracle-valid spans, so this is a bounded diagnostic rather than the primary
repair. A fixed-K near-terminal/tied-ordering problem remains: macro
cost-to-go is invariant to within-span order, although strict success depends
on the first receding-horizon action. Follow-ups must report this separately
from transition reachability.

## Adaptive hierarchy result

The prefix-aware audit localized the apparent selector failure to the end of
the episode. On 400 oracle-trajectory anchors, the learned macro starts with a
necessary action with probability 0.99 at three remaining steps, 0.96 at four,
and 0.83--1.00 beyond four. It is 0.42 at two and necessarily 0 at one because
the current planner filters for full three-action spans. This is a fixed-option
termination error, not a general failure of the macro representation.

The learned primitive state-value head is therefore used as an option
termination rule: use the hierarchy while predicted remaining distance is
above a threshold, then switch to one-step value control near the goal. The
head has validation MAE 0.49; its means are 2.27 at true distance two and 3.11
at distance three. Results below use the same 100 strict-budget episodes.

| Planner | switch threshold | Strict | Macro decision rate | Distractor rate |
|---|---:|---:|---:|---:|
| Oracle-valid macro spans + learned lower controller | 2.00 | .74 | .72 | .087 |
| same | 2.50 | .79 | .56 | .075 |
| same | 2.75 | .81 | .48 | .075 |
| same | 3.00 | .81 | .40 | .082 |
| same | 3.25 | **.82** | .34 | .080 |
| HWM continuous CEM, fully learned | 2.00 | .42 | .79 | .259 |
| same | 2.75 | .58 | .52 | .167 |
| same | 3.00 | **.65** | .43 | .144 |
| same | 4.00 | .76 | .14 | .107 |
| unrestricted future text spans + learned support | 3.00 | .55 | .47 | .192 |

The oracle-valid discrete hierarchy is now within .02 of the flat value
planner (.84) while still making 34--48% of decisions at the macro level.
The fully learned continuous CEM hierarchy improves from .08 to .65 at the
balanced threshold. Returning the final CEM mean is better than returning the
best sampled code (.65 versus .63), and nearest-span projection is worse
(.59); retain the faithful HWM mean/variance update.

Dense open-loop supervision and endpoint-only lower repair both reduce the
recursive three-step low-model error from .580 to about .485. Their combined
model raises macro-to-span retrieval from .55 to .73; adding high-to-low
reachability raises it to .805 but was previously behaviorally harmful. The
active `intent_hvalue_full` cell combines dense open-loop loss, endpoint
repair, prefix-aware macro scoring, and the learned option-termination value.
It is the final test of whether better subgoal decoding closes the .82/.84 gap
and improves fully learned CEM without suppressing macro use.

That combined cell is now negative: its best oracle-valid result is .79 with
.35 macro use and its CEM result is .61. Improved rollout/retrieval metrics do
not translate to control and should not be retained. A separately trained
value head attached to the earlier macro checkpoint also remains .82/.61.

Further localization with the selected simple checkpoint gives .84 when the
true macro waypoint is paired with oracle-feasible lower search, but only .63
when the same waypoint is paired with unrestricted future text-span search.
Soft learned future-action support is not the solution: removing its penalty
improves the oracle-waypoint control from .63 to .70 and CEM from .65 to .66;
weights 0.1--10 and hard logit thresholds -2/0/2 do not improve further. A
lower terminal-value weight reaches .65-.66, but a zero-subgoal ablation gives
.64-.66, proving that this merely reverts to value-only control.

The active final lower-interface repair replaces open-loop future-action
enumeration with a goal-conditioned option policy
`Q_low(s, a, s_subgoal)`. It is trained listwise from valid counterfactual
macro spans and, at deployment, scores only the currently feasible intent
phrases before replanning. Two matched cells (`intent_hgoalpolicy_a16_w1/w4`)
test loss scaling; each will be evaluated with a true waypoint, enumerated
macro spans, and continuous 1200x20 CEM. This is the first lower controller in
the wave that is both closed-loop and does not require symbolic future action
availability.

## Goal-conditioned lower policy result

The first-action-only label was wrong for a set-valued macro subgoal: when
independent actions commute, any currently feasible action contained in the
target macro span is a valid first action. The corrected multi-positive
objective trains `Q_low(s,a,s_subgoal)` against this action set and against
both true and high-predictor subgoals. On the preserved epoch-2 checkpoint it
reaches .954 top-1 on true subgoals and .881 on predicted subgoals (random
.751 because each target has several valid actions).

On the original 100-episode screen, unrestricted discrete macros reach .82
and faithful HWM CEM reaches .80 at termination threshold 3.0. Threshold 2.5
keeps CEM performance at .79 while raising macro use to .50; threshold 3.25
gives .84 with .30 macro use. These small-screen values are diagnostic only.
The larger fixed test set gives the reportable result:

| Test seed | Flat one-step value | CEM hierarchy | Macro decision rate |
|---:|---:|---:|---:|
| 731 | .798 | .766 | .296 |
| 732 | .800 | .762 | .302 |
| 733 | .814 | .776 | .297 |
| **Mean** | **.804** | **.768** | **.298** |

Each row uses 500 strict-budget test problems, threshold 3.25, 1200 CEM
candidates, 20 refits, 10 elites, and variance EMA .9. The fully learned
hierarchy is therefore 3.6 points below its matched flat controller while
making 30% of decisions through the macro level. This replaces the previous
.65-.66 fully learned result. A 50%-macro operating point scores .754 on the
same seed-731 test set; use threshold 3.25 as the performance configuration
and 2.5 only for the hierarchy-utilization trade-off plot.

The exact-order control gives only .69 CEM/.60 true-waypoint success and is
rejected. The multi-positive epoch-14 checkpoint improves the auxiliary
true/predicted top-1 probe to .972/.966, but gives only .746 on the same
500-problem validation CEM control at threshold 3.25. Recalibration to 3.75
recovers .770 while reducing macro use to .197. The epoch-2 checkpoint gives
the same .770 with .286 macro use and therefore Pareto-dominates it. The
selected artifact is
`runs/intent_hgoalpolicy_set_a8_w1/selected_planning.pt` (epoch 2); do not
replace it with `best.pt` merely because the auxiliary loss is lower.

Post-selection support, reachability, ensemble, and value-ranking diagnostics
are recorded in `09_hierarchy_support_and_value.md`. None improves the
three-seed test mean; the selected artifact remains unchanged.
