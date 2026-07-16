# Wave 11 — controller outcomes and support-constrained discrete hierarchy

_Completed 2026-07-14. The apparent discrete gain was invalidated by an
action-order control, and the corrected support-guided planner failed its
predeclared paired confirmation. Hard-text and non-hierarchy experiments were
paused throughout this wave._

> **Correction.** The first discrete implementation truncated an exponentially
> expanding sequence list lexicographically. This over-represented early first
> actions—the exact actions executed by receding-horizon control. The H2/H3
> confirmations below reproduce that implementation and are not evidence for
> hierarchy. The planner now balances proposal count by first action,
> randomizes future proposals reproducibly, and reports an always-first
> baseline in every evaluation.

## Question

Why does continuous HWM-style macro-state planning fail despite accurate
high-level dynamics, and can hierarchy work when the planner is restricted to
macro actions obtained by encoding actual text-action sequences?

## Controller-outcome upper bound

For each high-level continuous CEM decision, the top 32 proposed subgoals are
executed for three closed-loop low-level steps. The candidates are then
reranked by their realized endpoint on the same 100-problem diagnostic split.

| top-32 reranking signal | strict success |
|---|---:|
| matched flat controller | .260 |
| ordinary continuous CEM | .260 |
| endpoint reachability residual | .260 |
| endpoint latent-goal distance, weight 1 | .280 |
| endpoint latent-goal distance, weight 10 | .300 |
| exact realized remaining work | **.440** |

The exact oracle reranker changes 46.24% of decisions, reduces mean remaining
work from 3.458 to 2.898, and reduces selected rollout distractors from .754 to
.192. The geometric endpoint residual does not improve. Thus the continuous
candidate set contains useful choices, but geometric reachability is not the
task-progress signal needed to find them.

## Learned controller-outcome heads

We collected 14,496 top-32 controller outcomes on data seed 41 and trained
regression, pairwise, and listwise heads to predict realized remaining work.

| learned head | held-out correlation | pair accuracy | top-1 optimal | regret |
|---|---:|---:|---:|---:|
| regression | **.794** | **.633** | **.736** | **.330** |
| pairwise | .747 | .611 | .670 | .396 |
| listwise | .609 | .592 | .692 | .385 |

On the held-out data-seed-42 control screen, flat and ordinary continuous CEM
score .31/.31. Regression, pairwise, and listwise reranking score
.26/.26/.29. Offline prediction quality therefore does not recover the exact
reranker's control gain. This branch is rejected as the primary next lever.

## Deployable discrete macro planner

The discrete planner changes the optimization domain:

1. Candidate sequences begin with an intent phrase feasible at the current
   state. Later actions are sampled from all unresolved problem actions; the
   planner does not receive their future symbolic feasibility.
2. Each action sequence is encoded by the trained macro-action encoder and
   recursively propagated by the high-level dynamics.
3. A learned state-conditioned action-support score penalizes implausible
   future sequences.
4. The planner directly executes the first currently feasible text action and
   replans after observing the new state.

This is the discrete text-span planning branch, not continuous latent-vector
CEM with a low-level subgoal controller. It uses the same current-state action
menu as the flat JEPA and intent-policy language-model baselines.

## Historical depth and support screen (invalidated)

On the held-out 100-problem data-seed-42 screen:

| planner | flat | H1 | H2 | H3 | H4 |
|---|---:|---:|---:|---:|---:|
| strict success | .310 | .180 | **.410** | .390 | .330 |

H2 support weights 0/.1/1/10 gave .40/.40/.41/.42. These rows all use the
unbalanced cap. Truncation becomes more severe with depth, so the curve cannot
separate planning depth from candidate-order bias and is excluded from model
selection.

## Historical H3 paired confirmation (invalidated)

Each row evaluates flat and deployable H3 discrete control on the same 500
newly generated 7–9-step problems.

| data / planner seed | flat | discrete H3 | delta | macro rate |
|---|---:|---:|---:|---:|
| 51 / 551 | .322 | .380 | +.058 | .614 |
| 52 / 552 | .312 | .396 | +.084 | .612 |
| 53 / 553 | .298 | .382 | +.084 | .606 |
| **mean** | **.311** | **.386** | **+.075** | **.611** |

Mean selected-action distractor rate appeared to fall from .194 to .155, but
all rows use the biased proposal cap. Replication across seeds cannot repair a
systematic protocol confound.

## Historical H2 paired confirmation (invalidated)

The frozen H2/support-weight-1 configuration confirms on three fresh paired
500-problem splits:

| data / planner seed | flat | discrete H2 | delta | macro rate |
|---|---:|---:|---:|---:|
| 61 / 661 | .304 | .332 | +.028 | .616 |
| 62 / 662 | .342 | .416 | +.074 | .612 |
| 63 / 663 | .332 | .380 | +.048 | .600 |
| **mean** | **.326** | **.376** | **+.050** | **.609** |

Mean distractor selection appeared to fall from .192 to .153. These results
are retained for provenance but are not used for selection.

## Corrected ordering audit

On the original 100-problem selection split, matched flat control is .310 and
the always-first policy is .100. Root-balanced arbitrary sequence planning
scores .110. A learned-support beam scores .150, and adding a unit-weight
one-step value safeguard reaches .250. Thus discrete planning remains below
the flat controller after removing the order bias.

On faithful iGSM, the reference parameter iteration order is itself a leak:
before menu randomization, always-first solves 100/100 long problems. The
interface now exposes a stable problem-specific shuffled action order to all
models. Earlier faithful planning numbers using reference order are invalid.

The next experiments fine-tune the support head on true, one-step-predicted,
and recursive-rollout states, then rebuild a root-balanced support beam. This
head is now complete. On the same 100-problem screen, support-only planning is
.100, adding one-step value at weight 1 gives .300, and weight 10 gives .320
versus .310 flat.

An oracle-future-feasible sequence diagnostic reaches only .170 under terminal
macro-state value despite selecting fully executable six-action plans with
4.7 necessary computations of progress. Adding the strong one-step value
raises it to .330. Prefix-aware macro action values with tie-break scales
.1/1/5 similarly give .170/.170/.150 alone and .330/.330/.320 with the
one-step safeguard. The remaining issue is first-action credit under
replanning, not merely OOD or infeasible proposals.

The best two-point screen was then frozen and evaluated on three fresh paired
500-problem sets:

| data / planner seed | flat | corrected hierarchy | delta | macro rate |
|---|---:|---:|---:|---:|
| 71 / 871 | .318 | .310 | -.008 | .626 |
| 72 / 872 | .332 | .326 | -.006 | .610 |
| 73 / 873 | .300 | .296 | -.004 | .626 |
| **mean** | **.317** | **.311** | **-.006** | **.621** |

The effect is consistently negative. The distractor rates of the hierarchical
controller are .205/.191/.199, but this does not translate into episode
success. The corrected hierarchy is therefore rejected as an addition to the
flat recipe and retained as a diagnostic result.

## Conclusions to date

1. Continuous latent-action optimization fails because learned scoring does
   not reliably identify task-progressing controller outcomes; the exact
   realized-progress oracle proves that headroom exists.
2. Restricting macro actions to encoded text sequences remains a useful
   interface hypothesis, but the corrected 3×500 confirmation is negative.
3. Lexicographic candidate caps and reference action ordering are decisive
   confounds; both are now guarded by explicit baselines and balanced search.
4. The easy-domain hierarchy stage is closed. Its predictive gains and control
   failure motivate testing semantic token-to-phrase-to-sentence temporal
   abstraction in the harder domain after the flat easy-domain paper matrix is
   completed.
