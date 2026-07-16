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
