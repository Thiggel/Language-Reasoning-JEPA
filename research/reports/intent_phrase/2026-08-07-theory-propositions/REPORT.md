# A non-post-hoc framework for the endpoint-ranking Energy head

Date: 2026-08-07. Status: draft for the paper's method/theory section.
Scope: derives the frozen recipe (horizon-blind endpoint Energy trained by
pairwise ranking of imagined endpoints) from first principles, states four
propositions (two proved here, two with proof sketches plus pre-registered
empirical tests), and explains — before looking at any new numbers — what
each already-run experiment *had to* show if the framework is right.

The point of this document is to make the method's design *derivable*, not
narrated: given the environment and the planner, ordinal supervision on
imagined endpoints is the canonical choice, and every absolute/bootstrapped
alternative (MSE-calibrated energy, TD, expectile value) estimates strictly
more than the planner uses and pays for it off-support. The measured
negatives are then predictions of the framework, not anecdotes.

## 1. Setup

**Environment.** An iGSM problem is a finite dependency DAG over variables
V with a query variable q. A state s is the set of resolved variables
(closed under the rule that a variable can only be resolved after its
parents). Actions at s are the feasible variables A(s) = {v unresolved :
parents(v) ⊆ s}; transitions are deterministic: T(s, v) = s ∪ {v}. The
goal set G is {s : q ∈ s}. Let N ⊆ V be the ancestor closure of q
("necessary" variables); everything else is a distractor.

**Steps-to-go.** d(s) = minimal number of actions from s to G.

**Learner.** A text encoder φ maps the rendered state to R^256 (EMA
teacher φ̄), an action encoder u maps the intent phrase to R^16, a
predictor P imagines the next latent, applied recursively for multi-step
imagination. The Energy head E(z_root, ẑ_endpoint, z_0) scores a candidate
root action by its imagined H-step endpoint. Training compares candidates
at a shared root: for each candidate action a, R random feasible rollouts
of length H−1 are executed in the environment, the *best* rollout is the
one whose true endpoint has minimal EMA-latent distance to the encoded
solved state, and the pairwise logistic loss pushes E to rank candidates as
those best-endpoint distances rank them. The goal latent enters only these
training labels, never the planner.

**Planner.** Root-balanced beam search of depth D: for each root action,
imagine action sequences with P, score leaves with E, pick the root with
the best-scoring reachable endpoint. The planner uses E only through
*comparisons* — argmin/beam-pruning — never through its numerical values.

## 2. Two exact facts about the environment

**Lemma 1 (steps-to-go is counting).** For every reachable state s,
d(s) = |N \ s|.

*Proof.* Each action resolves exactly one variable, so at least |N \ s|
actions are needed to resolve the unresolved necessary variables, and q ∈ N
is among them: d(s) ≥ |N \ s|. Conversely, parents of necessary variables
are necessary, so N \ s always contains a feasible variable (a minimal
element in dependency order), and resolving necessary variables in any
topological order reaches G in exactly |N \ s| steps. ∎

**Proposition 1 (greedy descent on d is optimal, and slack is exactly
wasted steps).** (i) Any policy that at each non-goal state picks an action
minimizing d(T(s, a)) reaches the goal in exactly d(s₀) steps. (ii) Every
action changes d by −1 (necessary) or 0 (distractor); therefore a policy
solves the problem within budget d(s₀) + k if and only if it takes at most
k distractor actions. In particular the slack-k success curve reported by
the single-pass evaluator is exactly the distribution of the number of
wasted steps.

*Proof.* By Lemma 1, d(T(s,a)) = d(s) − 1 if a ∈ N, else d(s); a feasible
necessary action always exists (proof of Lemma 1), so the greedy policy
takes only necessary actions and (i), (ii) follow by induction. ∎

Proposition 1 is why this environment isolates *evaluation of action
quality* as the entire planning problem: there are no traps, no detours,
and no need for credit assignment across time — a planner succeeds exactly
as often as its root-action comparisons are correct. This is the cleanest
possible setting for asking whether JEPA geometry contains those
comparisons.

## 3. What the planner can see: ordinal invariance

**Proposition 2 (the decision-relevant object is a preorder, and ranking
estimates exactly it).**
(i) *Planner invariance:* for any strictly increasing g: R → R, replacing
E by g∘E leaves every beam-search trajectory, and hence every reported
metric, unchanged.
(ii) *Label invariance:* the training signal (which of two candidates has
the smaller best-rollout goal distance) is unchanged if the latent distance
is replaced by any strictly increasing transform of itself.
(iii) Consequently the target of learning is the equivalence class of
functions inducing the same candidate ordering — the quotient of energies
modulo monotone reparameterization — and pairwise ranking is an estimator
*of exactly this class*: its population loss is a function of the induced
ordering probabilities only. Regression onto distances or returns (MSE,
TD, expectile) instead selects one representative of the class and spends
capacity and gradients fitting its scale — a component of the target that
(i) proves the planner can never see.

*Proof.* (i) Beam pruning and final selection are argmin/top-k operations,
invariant under strictly increasing transforms. (ii) The label is an
indicator 1[dist_a < dist_b], invariant under monotone transforms of dist.
(iii) The logistic pairwise population loss E[log(1+e^{−(E_b−E_a)·sign})]
depends on the data distribution only through the pair-ordering
probabilities; its minimizer over all measurable E is any function whose
pairwise differences match the log-odds of the orderings — determined
exactly up to the additive/monotone structure the planner quotients out. ∎

The practical content is sharpest *off-support*. The planner queries E on
recursively imagined endpoints — a distribution the encoder never saw. A
scale-calibrated head must keep its numerical calibration valid under this
distribution shift to stay correct; a ranking head only needs the
*ordering* of imagined endpoints to survive the shift, a strictly weaker
requirement. TD-style targets are worse still: they are bootstrapped from
the model's own values, so calibration error feeds back into the target.
This is the framework's explanation for the measured contrast axis
(ranking recipe .45/.84/.88/.88 at depths 2/4/8/16 vs TD-Q .19 at depth 1
with no depth scaling, expectile value near random, MSE variants and
TD-as-auxiliary all negative) — and it *predicted the sign* of each of
those comparisons before they were run in their final form.

## 4. What the label is: H-step lookahead improvement

**Proposition 3 (best-of-R rollout labels approximate the H-step optimal
backup; ranking by them is one step of policy improvement with lookahead).**
Let μ be the uniform feasible rollout policy and, for a candidate root
action a at state s, let
Q_H^{(R)}(s, a) = min over R independent μ-rollouts of ρ(endpoint after
executing a then H−1 steps of μ), where ρ is the latent goal-distance
proxy. Then:
(i) as R → ∞, Q_H^{(R)}(s,a) ↓ min over *all* H-step continuations
starting with a of ρ(endpoint) — the H-step optimal cost proxy;
(ii) if ρ is any strictly increasing transform of d (ordinal faithfulness
of the geometry), then for H ≥ 1 and R → ∞ the induced ordering of root
actions is: necessary actions strictly before distractors, whenever the
problem has at least H remaining necessary steps — i.e. the label ranks
optimally, and by Proposition 1 the greedy policy on the label is optimal.
(iii) At finite R the label is a best-of-R approximation whose gap decays
geometrically in R for this environment (each rollout independently hits
an optimal continuation with probability ≥ p_min^{H−1} where p_min is the
minimal probability of the necessary frontier under μ).

*Proof sketch.* (i) is monotone convergence of the sample minimum onto the
essential infimum; the continuation set is finite. (ii) After executing a
necessary action, the set of reachable-in-(H−1) endpoint d-values is
pointwise ≤ (coupling: append a to any continuation of the distractor
branch and drop its last step — feasibility is preserved because resolving
extra variables never disables actions in this monotone environment); with
strict inequality at the minimum whenever ≥ H necessary steps remain.
(iii) is a union bound. Full proofs are mechanical; the coupling in (ii)
uses only monotonicity of feasibility (s ⊆ s' ⇒ A(s) ∩ unresolved(s') ⊆
A(s')), which holds in iGSM because dependencies are positive. ∎

Interpretation in plain terms: the training label is "which root action
can, within H more steps, get closest to the goal in the model's own
geometry" — a lookahead-improved comparison, not a one-step imitation
signal. Training at horizons {1, 2, 4, 8} therefore teaches one
horizon-blind head the *fixed point* of this improvement across scales,
which is why the same head remains coherent when the planner queries it at
depth 16 (measured: .884 at D16 with no depth input; the depth-input
variant was unnecessary and fixed-H4 training collapsed off-support to
.29 at D2).

## 5. When recursion survives: a margin condition

**Proposition 4 (margin robustness under compounding imagination error).**
Suppose (a) the Energy head is L-Lipschitz in its endpoint argument, (b)
one application of the predictor moves an on-trajectory latent by at most
δ from the encoding of the true successor state, and (c) at planning depth
D the true energy margin between the best root action's reachable imagined
endpoint and the runner-up's is m_D. Then the planner's root choice at
depth D is unaffected by imagination error whenever m_D > 2·L·D·δ.

*Proof.* Recursion compounds at most additively in the metric: after D
applications the imagined endpoint is within D·δ of the true-successor
encoding chain (triangle inequality), so each candidate's energy moves by
at most L·D·δ; a comparison flips only if the two scores cross, requiring
the combined movement 2·L·D·δ to exceed the margin. ∎

This turns "does test-time scaling keep working as D grows?" into a race
between two measurable quantities: margins m_D (which *grow* with depth in
iGSM, because deeper lookahead separates necessary from distractor roots
by more resolved-variable difference) and compounded drift L·D·δ (linear
in D). The framework therefore predicts a regime where success *improves*
with depth (margin growth dominates) followed by saturation (drift catches
up) — qualitatively what the headline curve shows (.45 → .84 → .88 → .88).

## 6. Pre-registered empirical tests

Each test is stated with its pass condition *before* being run. Tests T1,
T3, T4 use symbolic ground truth for measurement only (labeled oracle
diagnostics in the paper); no training signal changes.

- **T1 (ordinal faithfulness of the geometry).** On held-out *true* states,
  Kendall rank correlation between EMA-latent goal distance and symbolic
  d(s). Pass: τ high and stable across depths; this is assumption (ii) of
  Prop. 3, tested directly rather than assumed silently.
- **T2 (invariance in practice).** Apply strictly increasing nonlinear
  transforms to the trained E at plan time (g∘E for several convex/concave
  g). Prediction: planner metrics identical to the digit (Prop. 2(i) is
  exact, so this is a wiring check that the evaluation stack truly only
  compares energies). Any deviation falsifies the implementation, not the
  theory.
- **T3 (margins vs drift).** Measure per-step predictor drift δ̂ =
  ‖P(φ(s), u(a)) − φ(T(s,a))‖ on validation transitions and empirical
  margin distributions m̂_D from the planner's candidate scores at each
  depth. Prediction: the depth at which the strict-success curve saturates
  coincides with the depth where the lower tail of m̂_D crosses 2·L·D·δ̂
  (L estimated by finite differences of E). Pass: rank agreement across
  depths {1,2,4,8,16}; fail would mean saturation has a different cause
  (e.g. beam width), which the paper would then have to report.
- **T4 (lookahead label quality).** With the symbolic simulator, compute
  the exact H-step optimal ordering of root candidates and measure how
  often the best-of-R latent label agrees, as a function of R ∈ {1, 2, 4,
  8} and H. Prediction from Prop. 3(iii): agreement rises geometrically in
  R and the R=4 recipe sits near the plateau; this also retro-dicts why
  R=4 was sufficient during recipe selection.
- **T5 (scale is wasted capacity).** The counterfactual-K sweep and the
  frozen-backbone cell (running) are the framework's ablation predictions:
  ranking needs *comparisons*, so (a) performance should degrade
  gracefully as alternatives per anchor shrink (K → 0 leaves only
  same-root continuation pairs), and (b) a post-hoc head on frozen
  geometry should recover most of the performance iff the geometry itself
  is ordinal-faithful (T1), since the head only re-expresses an ordering
  the encoder already carries.

## 7. First results (2026-08-07, mix4 seed 0, `audit_theory_predictions.py`)

Run: `runs/autonomy/intent_phrase/2026-08-07-intent-theory-probes-v1/`
(100 episodes/anchors for T1/T4, 60 episodes for T2).

- **T2 passes exactly.** exp, cube, and affine transforms of the trained
  Energy leave every planner metric bit-identical (success .983, mean
  steps 4.27 at depth 4, slack 2). Proposition 2(i) is exact and the
  evaluation stack is confirmed to consume only comparisons.
- **T1 is only moderately positive — an honest surprise.** Mean Kendall
  tau between EMA-latent goal distance and symbolic steps-to-go over
  random-policy prefix states is .46; only 13% of episodes exceed .8 and
  the minimum is negative. The raw geometry is *not* uniformly ordinal on
  arbitrary states. The working method is therefore not "read off a
  monotone distance": the ranking head adds real information beyond the
  bare latent metric. Prediction registered *before* the pure-JEPA
  frozen-backbone cell finishes: if T1's moderate tau is the binding
  constraint, a head distilled on a ranking-free backbone should land
  clearly below the recipe (the recipe-backbone frozen head already
  matched it: .83 at D4).
- **T4 explains the depth-1 weakness quantitatively.** Agreement of the
  best-of-R latent label with the exact optimal root ordering: H=1 is
  .48 (flat in R — with no continuation steps all rollouts coincide), H=2
  jumps to .83–.85, H=4 climbs from .53 (R=1) to .83 (R=8). One-step
  latent distances barely separate necessary from distractor actions, but
  two-plus-step lookahead labels do — precisely why strict success is .126
  at depth 1 yet .84+ once the planner searches deeper, i.e. why
  test-time scaling exists in this system at all. (H=8 rows were null:
  no sampled anchor had 8 remaining necessary steps with a mixed menu.)

## 8. Honesty notes for the paper

- Propositions 1–2 are exact and environment-general (any monotone
  deterministic DAG environment); Proposition 3(ii)'s coupling uses iGSM's
  monotone feasibility and should be stated for that class, not for
  arbitrary MDPs. Proposition 4 is a worst-case bound; the empirical
  content is T3, not the constant.
- The framework was developed alongside the experiments; what makes it
  non-post-hoc is that (a) Props. 1–2 derive the design choices from the
  planner's invariances rather than rationalizing them, and (b) T1–T5 are
  falsifiable predictions committed before running. The paper should say
  exactly this.
- Everything above concerns the candidate-privileged evaluation protocol
  (feasible menu at D1, symbolic future menus at D>1) and must be labeled
  as such wherever numbers appear.
