# Wave 10 — dense recursive supervision and hierarchy confirmation

_Completed 2026-07-14 17:28 CEST. Hard-text and non-hierarchy experiments
remained paused._

> Superseded for planner selection by Wave 11. The negative result below is
> specific to continuous latent-action CEM with low-level subgoal refinement;
> it does not apply to the subsequently confirmed discrete text-span planner.

## Question

Can dense recursive supervision at both hierarchy levels, a calibrated
macro-state value, and a goal-conditioned lower controller turn the
observed-intent hierarchy into a reproducible improvement over matched flat
planning on 7–9-step problems?

## Protocol

- Macro stride: 3 primitive actions.
- High-level planner: CEM with 1,200 candidates, 20 refits, 10 elites, and
  variance EMA 0.9.
- Execution: the first predicted macro state is passed to a goal-conditioned
  lower policy that scores only actions feasible in the current state; one
  primitive action is executed before replanning.
- Support: learned soft support weight 1 and conditional density weight 0.1.
- Option termination: switch to flat control below predicted remaining cost
  3.5.
- Selection screens use 100 or 200 7–9-step validation problems. Confirmation
  uses three independently generated sets of 500 paired flat/hierarchy
  episodes.

The encoded solved-state distance is called the HWM-style latent-goal
diagnostic. It requires the solved trace as an oracle goal observation, which
is unavailable when the answer is unknown. Exact remaining graph distance is
also explicitly oracle supervision when it trains a value. No deployed
hierarchy receives future feasible-action sequences.

## Dense high-level recursive supervision

The high predictor is applied recursively from every valid macro origin and
supervised at every intermediate horizon. LN–L1 error is:

| high supervised depth | h=1 | h=2 | h=3 |
|---:|---:|---:|---:|
| none | .5859 | .6285 | .6842 |
| 1 | **.5569** | .5995 | .6596 |
| 2 | .5585 | .5800 | .6307 |
| 3 | .5599 | **.5784** | **.6223** |

Depth 3 is selected for hierarchical planning because it has the best
multi-macro recursion, although depth 1 remains best at one step.

## Dense low-level recursive supervision

The same shifted-sequence objective is applied at every primitive origin.

| low supervised depth | h=1 | h=4 | h=8 | strict flat, n=200 |
|---:|---:|---:|---:|---:|
| none | .4930 | .6371 | .8674 | .300 |
| 1 | **.4698** | .5899 | .7069 | .335 |
| 4 | .4766 | **.5525** | .6138 | **.355** |
| 8 | .4804 | .5527 | **.6041** | .345 |

Depth 4 is the behavioral selection. Depth 8 has the best longest-horizon
error but does not improve flat planning further.

## Macro-state value calibration

At 200 held-out states, all 11,659 valid alternative three-action spans are
encoded and ranked. Exact remaining computations supervise the learned value
rows.

| high-level score | corr. | pair acc. | top-1 optimal | regret |
|---|---:|---:|---:|---:|
| terminal latent distance (oracle goal observation) | .478 | .591 | .590 | .545 |
| macro action value `Q(z,m)` | .727 | .694 | .715 | .405 |
| state value `V(z')`, factual | .857 | .729 | .790 | .295 |
| `V(z')` + counterfactual regression | **.860** | .734 | **.800** | **.280** |
| `V(z')` + regression + ranking | .859 | **.735** | **.800** | **.280** |

Next-state value is much better calibrated than latent terminal distance.
Counterfactual regression contributes a small gain; the added ranking hinge
does not improve top-1 or regret. A learned support classifier separates valid
codes from 3-sigma perturbations almost perfectly, but this does not establish
that it identifies task-useful actions.

## Lower-policy and measured-reranking diagnostics

Adding predicted-subgoal supervision to the lower policy slightly improves
its predicted-waypoint probe, but its best small planning result is only .28
and does not survive larger evaluation. Even a true encoded waypoint reaches
the flat success ceiling rather than exceeding it once option termination is
tuned.

A planning-time controller rollout was also used to rerank final high-level
candidates. On the 100-problem diagnostic:

| top-candidate selection | strict |
|---|---:|
| near-zero-weight top-8 control | .22 |
| measured endpoint reachability residual | .23 |
| measured endpoint-to-latent-goal distance | .27 |
| exact remaining graph distance after rollout | .29 |

The exact row changes the selected macro on 39% of decisions, lowers mean
remaining cost from 4.52 to 4.08, and lowers distractor selection from .704 to
.259. It is non-deployable and only reaches the flat ceiling. Endpoint
reachability alone is not a useful task-progress score.

## Planning-depth screen

On the 100-problem selection split:

| high-level score | H=1 | H=2 | H=3 |
|---|---:|---:|---:|
| learned macro-state value | .26 | .30 | **.33** |
| terminal latent distance (oracle goal observation) | .31 | .31 | .31 |
| matched flat | .28 | .28 | .28 |

Dense high-level recursion plus the learned state value gives the first
monotone hierarchy-depth curve. The HWM-style latent-goal score does not. This
is only a selection screen, not a positive result.

Auxiliary path-value weights 0.1/0.3/1 give .24/.29/.28 and macro-action-value
weights 0.1/0.3/1 give .28/.27/.25. Neither improves the plain terminal state
value. Hard projection, prior-noise CEM, support weights above 1, and removing
density regularization are also negative.

## Three-set confirmation

The selected checkpoint is `runs/intent_lodense_tail79_n4/best.pt`. Each row
uses paired flat and hierarchical control on the same 500 newly generated
problems.

| data / planner seed | flat | hierarchy | delta | macro rate |
|---|---:|---:|---:|---:|
| 31 / 731 | .314 | .272 | -.042 | .633 |
| 32 / 732 | .336 | .320 | -.016 | .623 |
| 33 / 733 | .322 | .274 | -.048 | .634 |
| **mean** | **.324** | **.289** | **-.035** | **.630** |

The apparent .33 versus .28 hierarchy gain on the 100-problem screen is
selection noise. The confirmed hierarchy is 3.5 points worse and selects more
distractors. It is rejected and must not be transferred to official iGSM.

## Conclusions

1. Dense recursive supervision works as an optimization intervention and
   improves both long-horizon dynamics and flat primitive planning.
2. Exact macro-state value distillation is substantially better calibrated
   than encoded terminal distance; pairwise ranking adds little after
   counterfactual regression.
3. CEM under-budgeting is not the explanation: the planner already uses the
   requested HWM-scale regime and its internal objective converges.
4. Off-support action search is observable but not the sole remaining gate:
   action banks, projection, conditional priors, GMMs, learned support, and
   epistemic penalties all fail to produce a confirmed gain.
5. The unresolved interface is controller-specific executability and realized
   progress. Open-loop endpoint proximity is not an adequate proxy for what
   the closed-loop lower policy will achieve.

## Next experiment

Train `R_pi(z,z_subgoal)` from actual K-step closed-loop lower-policy outcomes
and train the high-level task value on the realized endpoint rather than the
intended macro endpoint. First use exact rollout labels to measure the
available upper bound, then learn the two heads and rerank only the top CEM
candidates. Accept a hierarchy only after a fresh paired 3×500 confirmation.
