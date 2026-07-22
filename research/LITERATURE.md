# Decision-relevant literature ledger

Record narrow searches that change an experimental decision. Include query,
date, primary sources, applicable claim, and limitations. Do not build an
undirected bibliography.

Existing conceptual anchors are I-JEPA, HWM, Delta-JEPA, variational JEPA,
VICReg, and SIGReg. Before using a method as a baseline or making a novelty
claim, verify the primary source and current publication status.

## 2026-07-16 — transfer benchmarks and anticipated review concerns

- Query: language reasoning environments with explicit actions, outcomes, and
  controllable compositional/OOD evaluation.
- Primary sources:
  - ProofWriter: <https://arxiv.org/abs/2012.13048>
  - GSM-Symbolic: <https://arxiv.org/abs/2410.05229>
  - Tree of Thoughts / Game of 24: <https://arxiv.org/abs/2305.10601>
  - TextWorld: <https://arxiv.org/abs/1806.11532>
  - ALFWorld: <https://arxiv.org/abs/2010.03768>
  - CFQ: <https://openreview.net/forum?id=SygcCnNKwr>
  - ICLR 2026 reviewer guide: <https://iclr.cc/Conferences/2026/ReviewerGuide>
- Applicable claims:
  - ProofWriter is the best near-term second-domain candidate: rule applications
    can be exposed as observed natural-language actions, their inferred facts as
    outcomes, and proof completion as the goal. This preserves the scientific
    interface while changing the underlying reasoning algebra.
  - Game of 24 is a cheap planning stress test with explicit operations, but it
    is another small synthetic arithmetic domain and is insufficient by itself
    to establish language-domain generality.
  - GSM-Symbolic motivates controlled perturbations of numbers, templates, and
    irrelevant information. CFQ motivates systematic held-out compositional
    splits. These are evaluation-design precedents, not drop-in action datasets.
  - TextWorld/ALFWorld offer stronger external validity but substantially change
    action semantics, observability, and reward structure; use them only if the
    claim expands from language reasoning to general text-world modeling.
  - Review-facing priorities are technical attribution, fair baselines,
    experimental rigor, reproducibility, clarity, and a precise novelty claim.
- Limitations:
  - Converting ProofWriter into an action environment is a new benchmark
    construction and must prevent symbolic feasible-action filtering from
    becoming an unreported oracle.
  - Template datasets may reward lexical shortcuts. Every proposed transfer
    benchmark therefore needs paraphrase, identifier, and compositional controls.
- Design change:
  - Freeze the selected intent-phrase recipe before transfer. Run faithful iGSM
    plus ProofWriter as the minimum cross-domain package, with Game of 24 only as
    an inexpensive supplementary stress test.
  - Report strict success, tolerance-based success, and path regret separately;
    never describe a tolerance score as exact accuracy.
  - Add a direct preference/policy model trained from precisely the same GAR
    supervision to isolate whether predictive latent learning contributes beyond
    the ranking objective.

## 2026-07-22 — VISReg as a heuristic-free TextJEPA target stabilizer

- Query: can VISReg faithfully replace EMA targets and VICReg in the pooled
  token-action JEPA, and which computation does that actually remove?
- Primary sources:
  - Paper: <https://arxiv.org/abs/2606.02572>
  - Official implementation: <https://github.com/HaiyuWu/visreg>
- Applicable claims:
  - VISReg uses center and per-dimension scale penalties plus a
    sliced-Wasserstein shape loss on random projections. Its official
    ImageNet configuration uses 4,096 projections and weights regularization
    by 0.9 versus 0.1 for the invariance term.
  - The method is explicitly trained without EMA, a teacher network, or
    stop-gradient. For temporal TextJEPA this maps to reusing future positions
    from the same online causal encoder as prediction targets, allowing the
    ordinary target-encoder pass to be removed.
- Limitations:
  - The evidence is from multi-view vision, not causal language trajectories.
    TextJEPA has highly correlated token-position samples and additional GAR
    counterfactual encodings, so stability and speed cannot be assumed.
  - Exact GAR outcome labels still require encoding modified prefixes. VISReg
    removes the standard EMA pass, not those auxiliary counterfactual passes;
    an end-to-end timing comparison is therefore required.
- Design change:
  - Add a faithful online-gradient VISReg mode with 4,096 projections. Compare
    it against EMA+VICReg using matched full-objective steps, collapse/effective
    rank diagnostics, and a two-A100 timing smoke before any long replacement.
