# Decision-relevant literature ledger

Record narrow searches that change an experimental decision. Include query,
date, primary sources, applicable claim, and limitations. Do not build an
undirected bibliography.

Existing conceptual anchors are I-JEPA, HWM, Delta-JEPA, variational JEPA,
VICReg, and SIGReg. Before using a method as a baseline or making a novelty
claim, verify the primary source and current publication status.

## 2026-07-17 — faithful LDAD for structured text edits

- Query: which representation enters Delta-JEPA's action decoder, whether its
  encoder is stopped/EMA, and how multi-step decoding is conditioned.
- Primary source:
  - Delta-JEPA v1: <https://arxiv.org/abs/2606.31232>
- Applicable claims:
  - LDAD receives only the online latent displacement `z(t+1)-z(t)` and
    reconstructs the externally observed raw action. It does not receive
    concatenated endpoints or a learned action embedding.
  - The published base method jointly optimizes latent next-state prediction
    and action reconstruction end to end. It does not use EMA, stop-gradient,
    VICReg, or another distribution regularizer.
  - Its multi-step decoder receives `z(t+H)-z(t)`, uses learned action queries,
    and injects the displacement through adaptive normalization.
  - The main experiments use action-reconstruction weight `10`.  The paper's
    Push-T sensitivity sweep is `{0, 0.1, 1, 10, 20, 50, 100, 1000}`; its best
    reported point is `50`, while very weak and excessively strong weights are
    worse.  These values multiply continuous-action mean-squared error and are
    therefore anchors, not directly calibrated text-token coefficients.
- Limitations:
  - The paper studies continuous robot controls and MSE action reconstruction.
    Text edits have discrete operation, pointer, and token fields. Complete
    observed-action token cross-entropy is an explicit modality adaptation,
    not a literal reproduction of continuous-action MSE.
  - The paper does not establish that LDAD combines beneficially with EMA or
    VICReg; those combinations require a factorial here.
- Design change:
  - Keep the text LDAD input strictly to the online state displacement and
    decode the complete observed edit action. Compare EMA, EMA+VICReg, and
    EMA+VICReg+LDAD with identical zero-dropout dynamics.
  - Screen text LDAD at `{1, 10, 20}` plus the matched zero control.  Calibrate
    VICReg separately at `{0.02, 0.1, 0.5, 1.0}` because its normalized
    variance/covariance loss has a different scale from token cross-entropy.
  - Force EMA modules to remain in evaluation mode even when the parent model
    enters training mode. Treat LDAD accuracy as a health diagnostic; retain
    it only if transition or recursive planning metrics improve.

## 2026-07-17 — counterfactual coverage for faithful token edits

- Query: how much alternative-action data an offline action-conditioned world
  model should receive, and when synthetic counterfactuals become unsafe.
- Primary sources:
  - MoCoDA: <https://openreview.net/forum?id=okFF_tsUGZi>
  - The Edge-of-Reach Problem in Offline Model-Based Reinforcement Learning:
    <https://openreview.net/forum?id=3dn1hINA6o>
  - Budgeting Counterfactual for Offline RL:
    <https://openreview.net/forum?id=1MUxtSBUox>
- Applicable claims:
  - MoCoDA supports controlling the augmented state-action distribution rather
    than treating arbitrary synthetic alternatives as interchangeable. Its
    positive results rely on known or learned local factorization, which the
    current text-buffer model has not established.
  - Edge-of-Reach shows that even accurate learned dynamics do not make
    model-generated off-support rollouts automatically safe. Edit experiments
    must measure empirical action support and separate observed exact outcomes
    from recursively imagined outcomes.
  - Budgeting Counterfactual argues that counterfactual decisions should be
    bounded because extrapolation error compounds with horizon. This motivates
    a small alternatives-per-state screen rather than assuming more candidates
    are always better.
- Limitations:
  - These papers study offline reinforcement learning, not reconstruction-free
    latent prediction over text edits. None supplies a transferable numerical
    counterfactual ratio.
  - Faithful token edits have exact executable alternative outcomes during data
    construction, while target-derived edit quality and the clean terminal
    buffer remain candidate-privileged supervision.
- Design change:
  - Do not import a fixed ratio. Ablate `K={0,1,4,8}` exact alternative edits
    per visited state and report performance per unique problem, exact
    transition, optimizer update, and GPU-hour.
  - Separate mechanical counterfactual dynamics from preference supervision.
    Cross unique-anchor count against alternatives per anchor at approximately
    matched total transition exposure, and include shuffled-outcome plus
    on-support/hard-negative composition controls.
  - Keep recursive model-generated alternatives out of the first screen; add
    them only after exact-outcome dynamics and support gates pass.

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
