# Wave 09 — macro support, lower feedback, and value ranking

_Completed 2026-07-14. Hard-text experiments remained paused._

## Protocol

All behavioral screens preserve
`runs/intent_hgoalpolicy_set_a8_w1/selected_planning.pt`, its
goal-conditioned lower controller, and the option switch at predicted
remaining distance 3.25. Unless stated otherwise, results use the same 200
validation problems (seed 321), strict budget, one macro step, 1200 CEM
candidates, 20 refits, 10 elites, variance EMA .9, conditional-prior weight
.1, and learned-support weight 1. The matched flat controller scores .815.

The published HWM planner does **not** use a learned value function. It
optimizes macro actions by latent distance between the predicted terminal
state and the encoded goal, then gives the first predicted state to the lower
planner. This repo additionally tests a distilled state cost `V_hi(z')` and
macro-action cost `Q_hi(z,u)`. “Oracle goal” below means distance to the
EMA-encoded solved state; it is not symbolic graph distance. Exact remaining
necessary steps supervise the learned value extensions only.
All three controls retain the easy-domain goal-conditioned lower controller;
therefore the latent-goal row isolates the faithful HWM **high-level
objective**, not a fully policy-free reproduction of the paper.

## High-level task cost

| High-level cost | Strict | Macro rate | Distractor rate |
|---|---:|---:|---:|
| Distilled `Q_hi(z,u)` | **.795** | .306 | .093 |
| Distilled `V_hi(z')` | .780 | .309 | .091 |
| HWM latent terminal-goal distance | .760 | .308 | .103 |
| Matched flat value controller | .815 | .000 | .081 |

At two recursively predicted macro steps, both `V_hi` and latent-goal
distance score .755. A deeper high-level rollout therefore does not improve
this short domain yet.

## A. Restricting macro-action support

| Support mechanism | Strict | Interpretation |
|---|---:|---|
| Standard code-space CEM | **.795** | selected control |
| CEM over conditional-prior base noise | .795 | smooth `u=T(z,eta)` constraint |
| Exact problem-local encoded macro bank | .780 | actual observed-span support |
| Project final CEM code to nearest bank item | .775 | hard first-code projection |
| Local conditional full-covariance GMM, weight .1 | .760 | 4 components |
| Local conditional full-covariance GMM, weight 1 | .755 | 4 components |

The learned support classifier has a narrow useful operating point:

| Support weight | 0 | .1 | 1 | 10 |
|---:|---:|---:|---:|---:|
| Strict | .740 | .740 | **.795** | .775 |

Off-support control matters, but hard support restriction is not the answer.
The learned soft boundary can retain useful interpolated codes that exact
retrieval, projection, and a fitted mixture reject.

## B. Lower-level reachability feedback

The cheap reachability approximation enumerates problem-local K-step spans
whose first action is currently feasible, rolls them through the low model,
and penalizes the squared distance from a proposed high subgoal to its nearest
predicted endpoint. It was tested inside every CEM update and as top-32
post-hoc reranking.

| Integration | weight .1 | weight 1 | weight 10 |
|---|---:|---:|---:|
| Joint high-level objective | .785 | .770 | .755 |
| Top-32 reranking | .775 | .785 | .765 |

Both variants are negative. “Near an endpoint reachable by some open-loop
lower sequence” is not aligned with “the learned closed-loop lower controller
will execute this subgoal correctly.” Retain this as a diagnostic, not a
planner cost.

## C. Epistemic disagreement

Three high predictors were independently reset and trained while keeping the
selected state encoder, macro encoder, values, support model, and lower
controller frozen and aligned. On 9,385 valid macro candidates:

- mean valid disagreement: .0216;
- ensemble-mean prediction L1: .5629;
- disagreement/error correlation: .331;
- perturbation-detection pair accuracy: about .90;
- mean disagreement rises to .0564 at three empirical code standard
  deviations.

The signal detects OOD macro codes, but adding it to planning does not help:

| Epistemic weight | 1 | 10 | 50 |
|---:|---:|---:|---:|
| Strict | .775 | .785 | .780 |

Optimizer exploitation is therefore observable but not the dominant residual
behavioral error.

## Macro-value loss screen

Each cell reloads the selected checkpoint, reinitializes only `Q_hi`, freezes
all other parameters, and trains for the same five-epoch/20k-example budget.
Alternatives are valid low-level action spans from the same state. For a fixed
state, ranking remaining cost-to-go is equivalent to ranking macro advantage;
the prefix-aware loss additionally breaks terminal ties using earlier
progress.

| `Q_hi` supervision | Pair acc. | Top-1 optimal | Regret | Strict (val-200) |
|---|---:|---:|---:|---:|
| Selected mixed head | .674 | .910 | .125 | .795 |
| Scalar remaining-cost regression | .684 | .895 | .150 | .795 |
| Pairwise advantage hinge only | **.735** | .905 | .120 | **.805** |
| Listwise top-1 only | .684 | .900 | .145 | .740 |
| Prefix-aware receding value + rank | .662 | .870 | .180 | .785 |
| Regression + pairwise + prefix-aware | .685 | .905 | .115 | .790 |
| Regression + pairwise + OOD-value rank | .730 | **.940** | **.065** | .790 |

The OOD-value loss directly requires a perturbed code to have cost at least
one above its source code. It succeeds offline: 99.9% of 3-sigma perturbations
receive worse Q, with mean margin 2.39. This does not improve planning.

The apparent pairwise-only validation gain also fails confirmation:

| Test seed | Selected mixed head | Pairwise-only head |
|---:|---:|---:|
| 731 | .766 | .764 |
| 732 | .762 | .760 |
| 733 | .776 | .768 |
| **Mean** | **.768** | **.764** |

Do not replace the selected checkpoint.

## Conclusion and next experiment

None of bank restriction, prior-noise search, local GMM density, lower-model
endpoint feedback, ensemble disagreement, deeper high-level rollout, or a new
value-ranking loss closes the selected model's .768 versus .804 test gap.
The main new insight is negative but localized: macro support is necessary,
yet visible off-support search is not the remaining bottleneck.

The next hierarchy experiment should train and score **closed-loop subgoal
executability**, not open-loop endpoint proximity. Generate CEM subgoals,
roll the frozen goal-conditioned lower controller for K receding steps, and
train `R_pi(z, z_subgoal)` to predict achieved residual/success. Rerank only
the top high-level candidates with this calibrated controller-specific score.
Run this first on a longer stylized split where at least two macro decisions
are routinely required; the present test averages only four necessary steps
and invokes the macro level on about 30% of decisions, so H=2 has little room
to help. Keep all Wave-09 mechanisms off by default.
