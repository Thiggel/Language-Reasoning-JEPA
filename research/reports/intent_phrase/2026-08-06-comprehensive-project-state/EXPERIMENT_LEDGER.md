# Intent-phrase experiment ledger

This ledger complements `REPORT.md`. It records the complete experiment programme by scientific purpose rather than treating every failed resubmission as a separate scientific condition. Raw process history remains under `runs/autonomy/intent_phrase/`; 147 top-level controller rounds and 356 retrieved `metrics.json` files existed at the 6 August audit.

## Coverage rule

- A retry with unchanged scientific configuration is represented by its latest complete artifact.
- Failed, timed-out, cancelled, unknown, and partial submissions remain in controller history but are not interpreted as negative model results.
- A changed seed is a replicate, not a new method.
- A changed evaluator, candidate interface, score composition, or leakage repair is a new scientific condition.
- The complete recovered D1/D4/D8/D16 tables appear in `REPORT.md`, Section 9a.

## Programme ledger

| Dates | Programme | What changed | Outcome |
|---|---|---|---|
| Jul 16-17 | Causal J3 context and optimization | Context 1, 4, full; LR and time-cap recovery | Context/optimization did not close action-selection gap |
| Jul 18 | GAR breadth and calibration | Counterfactual K, ranking/regression weights, finite-loss repair | Ranking useful; breadth saturates; calibration sensitive |
| Jul 18 | Dense recursion | Recursive depths 2, 4, 8 and weights | No reliable control gain; deeper jobs costly/unstable |
| Jul 18-20 | Hierarchy | Separate levels, macro bottlenecks, direct/residual, wide CEM | Corrected hierarchy negative; removed from paper method |
| Jul 18 | Width/LR proxy | Width 256, 384, 512 and LR checks | No clean scaling conclusion; superseded by fixed-budget study |
| Jul 20-21 | Learned catalogue prior | Phrase-pooled history, token history, global/prerequisite support | Token history improved feasibility; prior removed by paper steering |
| Jul 20 | Hybrid proposal plus JEPA rerank | Learned top-M proposal then JEPA valuation | Useful diagnostic, not final no-prior method |
| Jul 20-Aug 3 | Length evaluations | Prompt factorization, exact length, distractor-matched OOD | Initial curve confounded; corrected sentence curves degrade OOD |
| Jul 21-24 | Process/data gates | Schema, action support, shuffled controls, storage | Several admission and infrastructure bugs repaired |
| Jul 22-23 | ALFWorld | Adapter, branch-complete data, replay, overfit, causal prefix, counterfactual coverage | Transition memorization succeeds; planning admission fails |
| Jul 23 | ProofWriter | Compiler/executor, full catalogue, feasible diagnostic | Tiny feasible-only diagnostic 87.5% at slack four; full catalogue fails |
| Jul 23 | PlanBench-3 | Compiler/executor, shuffle, full catalogue | Tiny feasible-only diagnostic 50%; shuffled control 0% |
| Jul 23-30 | Faithful iGSM | Admission, invalid counterfactual repair, oracle fixes | Not promoted to paper-scale final method |
| Jul 29-Aug 3 | Stylized 300k paper baseline | Token LM, sentence LM, sentence+latent, MLP JEPA, causal JEPA | Five-seed main table established |
| Jul 31-Aug 3 | Representation analysis | Probes, effective rank, clustering, PCA exports | Non-collapse/progress evidence; qualitative minimal pairs missing |
| Aug 3 | GAR decision audit | Aligned action ranking, forced errors, teacher/predictor/head localization | Learned scorer much stronger than raw geometry |
| Aug 3-5 | GAR RQ3 controls | Direct ranker, detached geometry, geometry only, five-seed recoveries | Direct ranker matches one-seed ID; raw geometry weak |
| Aug 3 | Looped baselines | Poisson/loop protocols, local LR, loop count | Intermediate loop helps; FLOP tracer invalid |
| Aug 4 | First test-time scaling | Fixed checkpoints, symbolic sequences, direct ranker controls | Apparent near-perfect curve invalid due terminal/path metadata |
| Aug 4 | Depth failure audit | World-model drift, true-state scoring, oracle distance | Head semantics and search composition implicated, not drift alone |
| Aug 4-5 | Local Energy and GAR scoring | Terminal/cumulative scores, state/advantage targets, MSE weights, ranking losses | Local scores generally collapse with depth |
| Aug 5 | Planner repairs | Terminal score only, true beam, root balancing | Root-balanced terminal endpoint scoring wins |
| Aug 5 | Mixed-horizon endpoint | Mixed horizons, dense weights/discounts, frozen hybrid | Strong pre-safety curves; several initial archive failures recovered |
| Aug 5 | Terminal safety | Requested horizon, absorbing terminal padding | Five-seed scaling reproduced without terminal-length leak |
| Aug 5-6 | Full catalogue | No-op invalids, absorbing failure, balanced sampling | No-menu planning remains near zero |
| Aug 6 | Fixed-horizon simplification | H2/H4/H8/H16, R1/R2/R4/R8/R16, objective removals | H4 and R2 strongest; endpoint rank and counterfactual loss important |
| Aug 6 | Dense endpoint repair | Within-horizon rather than cross-horizon ranking | Scientifically repaired but still strongly negative |

## Results still absent

| Missing result | Current state |
|---|---|
| Coherent Energy support at every depth 1,2,4,8,16 | Not trained |
| Supported intermediate beam pruning | Not isolated |
| Clean fixed-H4 five-seed result | Not complete |
| Coherent endpoint direct-ranker control | Not trained |
| Endpoint geometry/action-order diagnostics | Not complete |
| Controlled paraphrase/negation/operator-swap set | Not complete |
| JEPA/token distractor-matched OOD five-seed curves | Not complete |
| Correct test-time FLOP curves | Tracer unresolved |
| Frozen endpoint recipe on faithful iGSM | Not complete |
| Paper-scale second domain | Not admitted |
| Full-catalogue learned planner | Current approaches fail |

## Superseded or excluded claims

- The early near-perfect deep-search curve is excluded because terminal sequence metadata leaked success.
- Hierarchical improvement is excluded after corrected negative confirmations.
- Action-prior improvements are not evidence for JEPA simulation and the prior is not in the paper method.
- Feasible-only ProofWriter/PlanBench results are candidate-privileged diagnostics, not unrestricted planning.
- ALFWorld transition memorization does not imply planning success.
- Effective rank proves non-collapse only; it does not explain control.
- PCA, t-SNE, and UMAP are not quantitative geometry evidence.
- Slack-four OOD results with only three distractors are ceilinged and uninformative.
