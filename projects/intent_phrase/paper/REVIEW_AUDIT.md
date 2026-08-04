# Internal reviewer audit

This audit records criticism applied during the 2026-08-04 narrative rewrite.
It is not part of the submitted manuscript.

## Round 1: theory and accessibility

Likely objections:

- A general POMDP opening would overpromise relative to deterministic iGSM.
- Predictive equivalence is standard and could be mistaken for claimed novelty.
- An arbitrary re-embedding argument ignores positive identifiability results.
- Information bottleneck language would be unsupported without a rate term.
- A geometric target is not automatically a Bellman advantage.

Revisions:

- The paper introduces the general controlled process, then immediately states
  the deterministic finite-horizon specialization and point-predictor caveat.
- Proposition 1 is described as standard in spirit. Its role is to define the
  required state.
- Proposition 2 excludes settings with additional identifiability assumptions.
  The controlled-world-model result up to orthogonal maps is cited.
- Quotient minimality carries the compression argument. Information bottleneck
  is background only and no compression claim is made for the model.
- The implemented horizon-two label is called a geometric progress surrogate.
  Exact control advantage is reserved for the sufficiency proposition.

## Round 2: implementation fidelity and causal attribution

Likely objections:

- GAR could be confused with one architectural score head.
- The notation could imply that the terminal solution embedding is available
  to the deployed policy.
- The gain could come from more counterfactual transitions rather than action
  ordering.
- A learned score can improve while predictive geometry remains unchanged.
- The no-GAR and latent-MSE-only rows appear contradictory unless their
  counterfactual supervision is stated.

Revisions:

- GAR now names the ordering supervision. Transition-value GAR, direct GAR,
  and raw-distance GAR name three realizations.
- The deployed score is `V(predicted_successor, problem_anchor)`. The terminal
  EMA state appears only in training labels. Raw terminal-distance planning is
  marked non-deployable.
- The central table exposes ranking, pairwise regression, and counterfactual
  successor prediction as separate columns.
- Detached-body, raw-distance, direct-GAR, shuffled-label, and successor-only
  controls are framed as required causal tests.
- The former no-GAR row is named factual prediction only. Latent prediction
  only includes factual and counterfactual successor losses.

## Round 3: empirical rigor, novelty, and presentation

Likely objections:

- The paper could read as a failed attempt to beat language-model baselines.
- Five seeds of 200 episodes do not support fine-grained model ordering.
- iGSM and a supplied feasible menu limit external validity.
- Direct GAR matching transition-value GAR weakens the factorization claim.
- Global effective rank and an uncontrolled t-SNE do not establish planning
  geometry.
- Current test-time curves do not isolate useful added computation.
- Recent value-guided, straightening, and temporal-distance JEPAs narrow the
  novelty claim.

Revisions:

- The abstract and introduction lead with the prediction-versus-ordering
  mechanism. The strongest sentence result is reported without apology.
- The manuscript requires paired episodes, larger evaluation sets, paired
  bootstrap intervals, and paired model differences before submission.
- Candidate privilege is a laboratory intervention. The limits section
  restricts the claim to conditional planning on stylized iGSM.
- Predictive factorization is unresolved. It must beat direct GAR on a
  preregistered generalization, data-efficiency, or valid simulation axis.
- Local ordering, regret, margins, and checkpoint correlations are primary.
  t-SNE and UMAP are withheld until controlled paraphrase and negation features
  exist. They will illustrate, not replace, full-space results.
- The current FLOP and deep-rollout curves are removed from the manuscript.
- Novelty is restricted to within-state counterfactual ordering over discrete
  language actions and its controlled factorization analysis.

## Remaining submission blockers

1. Complete five-seed gradient-routing and direct-GAR controls.
2. Run local action-order metrics across seeds and retained checkpoints.
3. Collect controlled paraphrase, negation, operator-swap, and rename features.
4. Add a validated non-arithmetic language-action domain.
5. Evaluate the random policy and all headline methods on the same larger
   paired episode set with paired confidence intervals.
6. Admit test-time scaling only after fixed-work FLOP tracing and stable
   recursive prediction pass.
