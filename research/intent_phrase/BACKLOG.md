# Observed-intent hierarchy and paper backlog

_Status 2026-07-14: Wave 11 invalidates the apparent discrete H2/H3 gain and
closes the corrected root-balanced planner as negative._

Dense recursive supervision improves high- and low-level rollout error, and
exact macro-state value distillation is better calibrated than latent goal
distance. Continuous latent-action CEM fails confirmation (.289 versus .324),
and learned controller-outcome reranking does not repair it. Restricting macro
actions to encoded text sequences initially appeared successful, but the
finite sequence bank used a first-action-biased lexicographic cap. Corrected
arbitrary-sequence H2 gives .110 versus .310 flat; learned-support beam plus a
one-step value safeguard reaches .250. Planning-matched support plus a strong
one-step safeguard reaches .320 on the selection split, but its frozen paired
3×500 result is .311 versus .317 flat. See
`waves/11_controller_outcomes_and_discrete_hierarchy.md`. Hard-text and
non-hierarchy work remain paused.

## Stage H0 - protocol and base (complete)

- Use stylized iGSM for fast screening and faithful iGSM for confirmation.
- Keep validation for selection and leave final test sealed.
- Hold the provisional non-symbolic deterministic recipe fixed while
  identifying hierarchy; do not resume fixed-point shearing.
- Report strict-budget accuracy first, +2-action accuracy second, then
  transition, value-order, subgoal-reachability, and compute diagnostics.

## Stage H1 - learn a faithful high-level model (complete)

Train shared-latent macro transitions from spans of observed intent actions.
The high-level prediction must be used at planning time, not merely added as a
training auxiliary.

The initial bottleneck screen is complete: deterministic `d_macro=32, K=3`
was best (.37 strict / .67 +2), but remains below the flat value planner
(.84). Counterfactual macro dynamics, exact state value, exact action value,
same-state ranking, and conditional support have now been implemented and
validated independently. Endpoint alignment was not the behavioral gate.
The successful interface uses learned option termination plus a
goal-conditioned closed-loop lower policy with a multi-positive action-set
target for commuting macro actions.

Original screen order (retained for provenance):

1. deterministic macro bottleneck size `d_macro in {2,4,8,16,32}` at stride 3;
2. stride `K in {2,3,4,6}` at the selected dimension;
3. deterministic encoder versus Gaussian posterior plus state-conditioned
   Gaussian prior at the selected `K,d_macro` (complete; deterministic won);
4. one high level versus two recursively composed high levels;
5. capacity-matched flat controls.

The probabilistic macro variant is useful only when it includes a calibrated
conditional density `p(m | s)`. Posterior variance alone is not an
off-manifold detector. Planning uses `-log p(m | s)` as an explicit support
penalty and reports it separately from goal value.

Train and report two distinct high-level costs:

- `V_hi(s')`, distilled from exact remaining graph distance after the span;
- `Q_hi(s,m)`, distilled from the exact macro advantage/cost-to-go, with a
  same-state pairwise ranking term.

The planner may use oracle terminal latent distance only as a labeled
diagnostic. The deployable scorer uses learned `V_hi` or `Q_hi`.

For sampled macro codes, use the learned conditional prior and a separate
support classifier. Positive support examples are valid low-action spans;
negative examples are state-shuffled codes and perturbations at multiple
empirical-code scales. Keep support separate from goal value.

## Stage H2 - hierarchical planning algorithms (not selected)

Every main comparison must actually invoke the high-level model.

1. **Prior shooting:** sample macro sequences from `p(m|s)`, unroll the
   high-level model, choose the lowest high-level value plus density cost, and
   pass its first predicted state as a subgoal to the one-step feasible-action
   selector.
2. **Prior-regularized CEM:** optimize continuous macro codes, updating the
   proposal from elite samples while retaining a conditional-prior penalty.
3. **Retrieved discrete macros:** retrieve on-manifold macro codes/action
   spans from training trajectories, plan over them, and use the first latent
   subgoal for low-level action selection.
4. **Oracle controls:** repeat 1-3 with oracle terminal distance and, only as
   a separate upper bound, oracle future feasible-action sequences.

Initial grid: high-level horizon `{1,2,4}`, density weight `{0,0.1,0.5,1}`,
and low-level latent-subgoal weight `{0.5,1,2}`. Report only HWM-scale CEM:
900--3000 high-level candidates, 15--40 refits, fixed elite counts, and
variance-EMA smoothing. A 256-sample/3-refit run is only an engineering smoke
and is never reported as HWM CEM. Compare algorithms at matched predictor
evaluations and time.

The previously selected continuous mechanism was
`runs/intent_hgoalpolicy_set_a8_w1/selected_planning.pt`: learned
distance-based option termination, 1200x20 HWM CEM, and a goal-conditioned
lower policy that scores only current feasible intent phrases. Across three
500-problem test seeds it gives .768 strict with .298 macro-decision rate,
versus .804 for the matched all-flat controller. This closes the historical
.07-.15 hierarchy failure to a 3.6-point gap. Before paper use, replicate the
selected training cell and the no-goal-policy ablation across three model
seeds; do not retune the threshold on test. Wave 10 subsequently adds dense
recursive supervision and a recalibrated state value, but its larger paired
confirmation remains negative (.289 versus .324 flat). The mechanism is
therefore not paper-ready. Wave 11 tests a support-constrained discrete
text-span planner, but its apparent H2/H3 gains are invalidated by
first-action-biased candidate truncation. Corrected root-balanced planning is
below flat control on all three confirmation pairs. No hierarchical planner is
selected for the main recipe.

## Stage H3 - hierarchy-aware dense open-loop supervision (initial sweep complete)

Let `S=E(x)` be the encoded latent sequence and `A` the aligned action
sequence. The causal predictor produces the complete shifted one-step
sequence

`S_hat^(1)[t+1] = P(S[t], A[t])`.

Feed the predicted shifted sequence back through the same causal predictor to
obtain `S_hat^(2)`, repeat to horizon `N`, and supervise every valid origin at
every horizon:

`L_N = sum_{i=1}^N lambda^(i-1) MSE(S_hat^(i)[i:], sg(S[i:]))`.

This is not the historical rollout from `s0` only. The initial high-level
`N={1,2,3}` and low-level `N={1,4,8}` sweep is complete at `lambda=1`.
High `N=3` lowers three-step error .684→.622; low `N=4` lowers eight-step
error .867→.614 and improves flat success .300→.355. The selected hierarchy
still fails confirmation, so weighting sweeps are deferred until
controller-specific executability is addressed.

## Stage H3b - controller-specific executability (complete; negative)

- Exact realized-progress reranking of the top 32 continuous candidates raises
  success .26→.44, proving candidate-set headroom.
- Learned regression/pairwise/listwise heads score .26/.26/.29 versus .31
  flat on held-out problems and do not recover the oracle gain.
- Do not repeat controller-outcome head tuning before faithful discrete
  transfer; the successful lever is the discrete action-support restriction.

## Stage H4 - remaining component screens

After H1-H3 freeze hierarchy/planning:

- regularizer factorial: EMA/online stop-gradient/fully online crossed with
  none/VICReg/SIGReg and faithful LDAD off/on;
- GAR horizon `{1,2,4,8,16}` using the selected high-level planner;
- temporal straightening in raw and projected geometry;
- state and value monotonicity;
- outcome anchoring, counterfactual outcomes, and residual/direct controls.

Do not assume deeper hierarchy, rollout, GAR, or planning will improve
monotonically. Plot the measured curves, including negative or plateauing
regions.

## Stage H5 - paper matrix, probes, and qualitative analysis

- Construct one cumulative build-up table from a minimal JEPA to the best
  selected configuration; place non-improving additions below a divider.
- Include random, token LM, sentence LM, and sentence+latent LM baselines.
- Run the selected model and final one-component ablations with three seeds.
- Run the complete representation-probe backlog in Wave 07.
- Select 3-5 matched episodes illustrating JEPA, token-LM, and sentence-LM
  differences without cherry-picking by outcome.
- Headline faithful iGSM; use stylized iGSM for controlled mechanisms and
  exhaustive audits.
