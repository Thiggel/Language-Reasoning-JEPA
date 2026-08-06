# Predictive planning with intent phrases: complete project state on 6 August 2026

## The one-sentence answer

The project has shown that a Joint-Embedding Predictive Architecture (JEPA) can choose language actions competitively in a controlled arithmetic environment, that action- or endpoint-ranking supervision is far more important than latent prediction error alone, and that recursive endpoint scoring can produce large depth gains, but the strongest scaling result still relies on symbolic future action menus and a horizon-conditioned Energy whose training and evaluation supports are not yet fully aligned.

## First, the idea in everyday language

Imagine a puzzle in which each card describes an operation such as “compute the number of blue objects from the number of red objects.” A conventional language model chooses the card whose wording seems most likely to come next. Our JEPA builds an internal vector for the current puzzle, imagines how that vector would change after one or more cards, and assigns a lower Energy to an imagined endpoint that appears more useful for finishing the puzzle.

The environment, not the model, executes the arithmetic and returns the resulting sentence. The model then observes that sentence and replans. This is therefore closed-loop action selection through language, not free-form generation of a complete chain of thought or numerical answer.

The project began with one-step Geometric Advantage Ranking (GAR). GAR orders candidate actions using the geometry of their predicted successor states. It then explored hierarchy, learned proposal mechanisms, full-catalogue action selection, causal Transformer predictors, multi-step latent losses, state values, action advantages, and several planning algorithms. The simplest currently promising method is different: recursively imagine a complete candidate sequence, then rank the final predicted endpoint with a learned lower-is-better Energy.

## Why this question matters

Autoregressive language modeling learns which text is likely to follow. Planning asks which action has useful consequences. JEPA offers a direct predictive factorization:

```text
current history + proposed language action
    -> predicted latent successor
    -> recursively predicted endpoint
    -> planning Energy
```

The theoretical motivation is that accurate latent prediction identifies controlled state transitions but does not determine a metric that must rank actions by goal progress. A reparameterization can preserve every predicted transition while reversing distances to a goal. Planning therefore needs an additional ordering constraint. Earlier experiments supplied it locally through GAR. The newer endpoint method supplies it over recursively imagined action sequences.

The scientific goal is not to claim state-of-the-art arithmetic reasoning. It is to test whether predictive representations can support explicit planning in language-mediated environments, identify which supervision creates useful planning geometry, and measure what JEPA representations preserve compared with generative language models.

## What we tested

### Task and interaction

The main development environment is stylized iGSM, a generator of arithmetic dependency puzzles. Training examples are fresh rather than a fixed memorized corpus. A typical episode is:

```text
prompt
 -> model chooses an intent phrase
 -> environment executes the operation
 -> environment returns an outcome sentence
 -> model observes the updated history and chooses again
```

The default paper-scale training budget used 30,000 new examples per epoch for 10 epochs, or 300,000 examples total, batch size 32, approximately 9,370 optimizer updates, learning rate 3e-4, 500 warmup updates, gradient clipping at 5, no dropout, and an exponential-moving-average target network whose momentum rises from about 0.99 to 0.999.

### Action interfaces

Three interfaces have appeared and must not be mixed:

| Interface | Real action candidates | Imagined future candidates | Scientific label |
|---|---|---|---|
| Current feasible menu | Symbolically executable actions | Not applicable at depth 1 | Information-matched controlled interface |
| Feasible future tree | Symbolically executable actions | Symbolic future feasible actions | Candidate-privileged for depth above 1 |
| Full catalogue | Every problem action, including invalid or repeated actions | Every action at every node | No feasible-action menu |

The strong deep-planning results use the feasible future tree. All compared endpoint candidates are scored by the learned model, but the environment graph supplies which future actions may be expanded. This isolates endpoint valuation and latent simulation; it does not solve action proposal or feasibility.

### Model families

The project trained or evaluated:

| Family | Decision mechanism |
|---|---|
| Random | Random action from the supplied candidate set |
| Token language model | Intent-token likelihood |
| Sentence language model | Next-intent likelihood |
| Sentence language model plus latent loss | Generative likelihood with auxiliary latent prediction |
| Looped language models | Weight-shared recurrent computation before scoring |
| Latent-MSE JEPA | Predict next latent state without geometric ranking |
| Local GAR JEPA | Predict one-step successor and rank actions with a geometric score/head |
| Direct ranker | Score history, goal context, and action without using the predicted successor as the decision bottleneck |
| Endpoint-Energy JEPA | Recursively predict a sequence endpoint and rank that endpoint |
| Causal-Transformer JEPA | Replace the MLP predictor with a causal Transformer predictor |
| Hierarchical JEPA | Add separately encoded macro levels and macro planning |

### Evidence levels

Results in this report have four different strengths:

| Level | Meaning |
|---|---|
| Five-seed result | Repeated training, usually reported as mean plus sample standard deviation |
| One-seed mechanism result | Useful for choosing a design, not a paper headline |
| Candidate-privileged diagnostic | Tests valuation or simulation with symbolic candidate support |
| Invalid result | Retained because it exposed a bug or confound, but excluded from claims |

## What a fair comparison means here

The five headline models received the same current symbolic feasible-action menu, prompt, and observed history. This is fair for comparing how models value a shared candidate set, but it does not establish unrestricted language-action planning. Deeper endpoint search additionally receives symbolic future feasible menus and is more privileged.

Strict success means solving in exactly the shortest number of valid actions. Slack-two success permits at most two excess actions. Existing evaluators usually ran separate strict and slack-two passes; a final evaluator should record first-solution time once and derive the complete slack curve from the same episode.

The headline five-seed table uses 200 validation episodes per seed. Most recent endpoint ablations use one seed and 300 episodes. Consequently, differences of a few percentage points among recent ablations should not be treated as stable rankings.

Several comparisons are only approximately capacity matched. The established five-seed models have about 17.7 million parameters for JEPA, 7.4 million for the token model, and 4.8 million for the sentence models. Hyperparameters were selected per family, but the entire predeclared dense learning-rate and scaling campaign was not completed. The paper should emphasize mechanism and controlled information access, not parameter-matched superiority.

## What happened

### 1. Main five-seed model comparison

These are the best established pre-endpoint results on stylized iGSM with the same current feasible-action menu.

| Model | Strict success | Slack two | Seeds | Main reading |
|---|---:|---:|---:|---|
| Sentence LM plus latent loss | **.521 +/- .047** | .809 +/- .020 | 5 | Best strict baseline |
| Sentence LM | .471 +/- .046 | .813 +/- .023 | 5 | Strong generative baseline |
| MLP GAR-JEPA | .433 +/- .019 | .796 +/- .033 | 5 | Competitive and stable |
| Token LM | .422 +/- .124 | **.836 +/- .073** | 5 | Similar mean, high variance |
| Causal-Transformer GAR-JEPA | .366 +/- .047 | .778 +/- .043 | 5 | Causal predictor did not improve this recipe |

The defensible result is viability. The JEPA is close to the generative baselines and substantially more stable than the token LM in this experiment, but the sentence-plus-latent model has the best strict mean.

### 2. Why local GAR was introduced

| Local-GAR condition | Strict | Slack two | Seeds | Interpretation |
|---|---:|---:|---:|---|
| Full MLP GAR-JEPA | .433 | .796 | 5 | Reference |
| Latent MSE only | .183 | .628 | 5 | Prediction alone is insufficient |
| No geometric ranking | about .077 | about .415 | Initially 2; later controls vary | Large collapse without ranking |
| Ranking only | .358 | not central | 5-seed campaign | Ranking carries much of the gain |
| No counterfactual latent prediction | .345 | not central | 5-seed campaign | Counterfactual dynamics also help |
| 1 counterfactual | .403 | — | 5 | Useful baseline breadth |
| 2 counterfactuals | .433 | — | 5 | Default |
| 4 counterfactuals | .439 | — | 5 | Small saturation gain |
| Advantage-MSE weight .10 | .443 | .802 | 5 | Nearby weight is similar to .25 |

This was the clearest early mechanism result: low next-latent error did not produce good action selection, while counterfactual ordering did.

### 3. GAR geometry and direct-ranker controls

The deployed GAR score on 500 aligned episodes reached top-1 optimal-action recall .786 and pairwise accuracy .826. Raw distance from the predicted next state to the goal representation reached only .549 top-1 and .603 pairwise accuracy. Thus the learned scoring surface was much stronger than raw global distance.

One-seed factorization controls found:

| Control | Strict | Reading |
|---|---:|---|
| Full local GAR-JEPA | about .433-.440 | Reference range |
| Direct ranker | .440 | Matches JEPA in-distribution on one seed |
| GAR gradients detached from body | .375 | Some evidence for useful joint shaping, not decisive |
| Raw geometry only | .190 | Euclidean/L1 goal distance alone is weak |

The direct-ranker result is an important unresolved control. In-distribution action selection does not yet prove that predicting successors is better than direct supervised ranking. The predictive factorization must earn its complexity through deeper search, data efficiency, compositional generalization, paraphrase robustness, novel-goal transfer, or another consequence-specific axis.

### 4. Test-time computation before endpoint Energy

Looped sentence models improved and then overthought. The original sentence model rose from .270 strict at one loop to .425 at four loops, then fell to .345 at sixteen loops; a learning-rate cross-check reached .475 at four loops. FLOP accounting was nearly flat across loop counts and is not trustworthy, so no FLOP-scaling comparison is ready.

An early JEPA deep-planning curve approached .97 but was invalid. The symbolic sequence enumerator returned solved paths early while the score included sequence length. The search procedure could therefore identify terminal paths through metadata instead of learned reasoning. This led to terminal-safe padding, fixed requested horizons, root-balanced beam search, and terminal-only scoring.

### 5. Terminal-safe mixed-horizon endpoint Energy

The first validated endpoint model sampled one training horizon from {1,2,4,8}, used four continuations per candidate root, ranked final endpoints, used no dense rollout loss, and kept requested horizon fixed after early success. Five seeds produced:

| Test depth | Strict mean | Strict SD | Slack-two mean | Interface |
|---:|---:|---:|---:|---|
| 1 | .120 | .012 | .560 | Current feasible menu |
| 4 | .834 | .018 | .990 | Symbolic future feasible tree |
| 8 | .874 | .029 | .991 | Symbolic future feasible tree |
| 16 | .877 | .029 | .995 | Symbolic future feasible tree |

This curve is terminal-safe and replicated. It is the strongest evidence that recursive endpoint evaluation can use extra test-time search. It nevertheless has a newly recognized semantic problem: the Energy head is trained at horizons 1, 2, 4, and 8, yet test depth 16 is outside support. More subtly, beam pruning at depth 4 uses partial depths such as 2 and 3, and depth 3 is outside the discrete training support. The head also receives horizon as an input. The result is therefore promising but not yet a clean final scaling claim.

### 6. Fixed-H4 endpoint Energy and recent simplification ablations

The fixed-H4 recipe uses one anchor, the factual root plus two feasible alternative roots, four sampled continuations per root, four recursive applications of the same residual MLP predictor, final endpoint pairwise ranking, factual and counterfactual one-step latent losses, chunk prediction, VICReg, and a weight-.25 root pair-difference auxiliary. Recent one-seed, 300-episode results are:

| Condition | D1 strict | D4 strict | D8 strict | D16 strict | D4 slack two |
|---|---:|---:|---:|---:|---:|
| Fixed H4, logistic ranking, R=4 | .107 in original matched run | **.910** | .913 | .897 | approximately 1.00 |
| Hinge ranking | .093 | .837 | .893 | .900 | .987 |
| R=1 continuation/root | .147 | .873 | .877 | .870 | .993 |
| R=2 | .110 | .903 | .913 | .907 | .987 |
| R=8 | .103 | .893 | .887 | .873 | .990 |
| R=16 | .117 | .880 | .883 | .867 | .980 |
| No root pair-difference auxiliary | .107 | .797 | .830 | .827 | .990 |
| No counterfactual latent prediction | .137 | .727 | .710 | .710 | .967 |
| No factual latent prediction | .110 | .860 | .853 | .847 | .990 |
| No endpoint ranking | .200 | .217 | .200 | .203 | .853 |
| No chunk prediction | .123 | .800 | .823 | .820 | .993 |
| No VICReg | .100 | .910 | .917 | .913 | 1.00 |
| Endpoint ranking only | .163 | .893 | .610 | .603 | .990 at D4 |

The `endpoint ranking only` row removes all auxiliary objectives rather than only one component. It is strong at exactly depth 4 and deteriorates away from that horizon. It demonstrates that endpoint ranking can nearly solve the candidate-privileged H4 task by itself, while auxiliary objectives appear to improve transfer across search depths.

The robust component conclusions from this one-seed screen are:

- Endpoint ranking is essential.
- Counterfactual next-state supervision contributes substantially.
- Two continuations per root may be sufficient; larger continuation counts do not help.
- VICReg appears removable in this setting, but effective-rank/collapse diagnostics and additional seeds are required before deleting it.
- Chunk prediction and factual one-step latent prediction provide smaller benefits.
- The root pair-difference auxiliary helps performance away from exactly H4, but it also gives the same Energy head different semantics at H1 and H4 and should not remain in the final coherent formulation.

### 7. Training-horizon ablation

| Fixed training horizon | D1 strict | D4 strict | D8 strict | D16 strict |
|---:|---:|---:|---:|---:|
| 2 | .120 | .807 | .770 | .770 |
| 4 | about .107 | **.910** | **.913** | **.907** in the matched R2 cell |
| 8 | .163 | .650 | .690 | .690 |
| 16 | .130 | .503 | .553 | .560 |

H4 is the best observed fixed teacher. Longer unrolls do not improve the method and likely combine harder endpoint ranking, greater predictor drift, and weaker optimization. This is not a monotonic training-horizon scaling law.

### 8. Dense endpoint and dense dynamics experiments

Two forms of dense supervision were explored. An initial implementation compared Energy values across different horizons, which is invalid because each horizon defines a different conditional prediction problem. A repaired version ranks candidates only within the same horizon. Even after repair, it performed poorly:

| Dense within-horizon model | D1 strict | D4 strict | D8 strict | D16 strict |
|---|---:|---:|---:|---:|
| H2 | .143 | .087 | .080 | .080 |
| H4 | .113 | .180 | .183 | .183 |
| H8 | .120 | .107 | .063 | .063 |
| H16 | .123 | .077 | .053 | .053 |

Earlier local-GAR rollout experiments also found that dense recursive losses generally reduced deep performance or increased variance. The likely explanation is not memory overhead alone. A single recursively shared predictor is being asked to make every intermediate prediction match an EMA state while the endpoint objective wants the full composition to preserve action ordering. Off-manifold recursive states, error correction, and one-step target matching can demand incompatible updates. This remains an inference; gradient-conflict and rollout-drift measurements are needed for a causal explanation.

### 9. Calibration and scoring-formulation matrix

Recent one-seed screens compared ranking, direct advantage/difference regression, direct state-energy regression, dense rollout supervision, and detached rollouts. The broad result is consistent:

| Formulation | Typical D1 strict | Typical D4-D16 strict | Conclusion |
|---|---:|---:|---|
| Pairwise/ranking local GAR | .36-.50 | .08-.22 | Strong one-step scorer, poor terminal rollout score |
| Direct advantage-difference MSE | .12-.19 | .01-.10 | Absolute calibration target is difficult |
| State-goal Energy MSE | about .14 | .10-.20 | Worse than ranking |
| Ranking plus direct MSE | about .12 | .04-.05 in direct-difference cells | MSE can overwhelm useful ordering |
| Dense or detached variants | mixed | generally low | No reliable repair |

Among alternative losses with four-step dense training, listwise ranking was often the least bad, but all were far below final-endpoint ranking. Hinge and logistic endpoint ranking are close enough to warrant a matched multi-seed comparison only after the method semantics are fixed.

### 10. Full-catalogue action selection

The no-menu experiment scores every action in the problem rather than receiving a symbolic feasible subset. With invalid actions treated as unchanged-state no-ops, strict success was essentially zero. Making invalid actions transition to an absorbing failure state and balancing valid/invalid sampling reduced some invalid selection but did not restore planning:

| Full-catalogue condition | D1 strict | D4 strict | D8 strict | D16 strict |
|---|---:|---:|---:|---:|
| No-op invalid, all sampling | .027 | .000 | .000 | .003 |
| No-op invalid, balanced | .013 | .003 | .010 | .007 |
| Absorbing failure, all sampling | .043 | .007 | .013 | .003 |
| Absorbing failure, balanced | .057 | .010 | .020 | .007 |

At D1, the best absorbing-failure run still used invalid actions around 20-24% of attempts and accumulated many valid distractors. At deeper search, failure did not disappear. More available actions are not harmless: many invalid/repeated/distractor candidates are weakly supported, invalid no-ops can appear neutral, and beam search creates many more opportunities for an erroneous low Energy. The action-embedding width of 16 may contribute, but the current evidence points first to supervision coverage and score extrapolation rather than embedding capacity.

The result means the current system is a candidate-valuing planner, not a complete proposal-and-feasibility system. A paper can use the shared feasible menu as a deliberate laboratory interface, provided every deep result is labeled candidate-privileged and full-catalogue failure is reported.

### 11. Hierarchy

The project explored separate higher-level encoders, macro bottlenecks, strides, recursive macro prediction, conditional priors/codebooks, wide Cross-Entropy Method search, iterative or supported planning, and dense lower-level rollout objectives. Corrected confirmations did not show a reliable planning gain. Several earlier positive hierarchy results were traced to planning/interface errors or oracle support. Hierarchy is therefore excluded from the paper-facing method and retained as a negative result.

### 12. Faithful iGSM and additional domains

The intended four-domain paper suite was faithful iGSM, ProofWriter, PlanBench Blocksworld, and text-only ALFWorld, with stylized iGSM as the mechanism environment. That suite is not paper-complete.

- Faithful iGSM development exposed candidate and counterfactual-coverage issues; no frozen final endpoint recipe has a validated five-seed headline result there.
- ProofWriter and PlanBench candidate-interface diagnostics showed large feasible-menu gains. On eight episodes, unchanged aligned checkpoints reached 87.5% ProofWriter and 50% PlanBench-3 success at slack four under the symbolic feasible subset, versus 0% and 12.5% for their best full-catalogue counterparts. PlanBench shuffled-action control remained 0%, supporting learned grounding rather than pure order leakage. These are tiny, candidate-privileged diagnostics.
- ALFWorld received adapters, deterministic replay checks, counterfactual collection, causal-prefix repairs, and memorization gates. The model memorized 115/115 within-episode transitions and repaired counterfactual full-state prediction reduced persistence error by 60-68%, but expert top-1 remained 11-12% and train success was 0/8. ALFWorld is not admitted to a paper-scale comparison.

### 13. Representation findings

Five-seed frozen-feature analysis used roughly 4,600 held-out states per seed:

| Model | Effective rank | Operation balanced accuracy | Remaining-step R-squared | Resolved-count R-squared |
|---|---:|---:|---:|---:|
| Sentence LM plus latent | 71.4 +/- 1.2 | **.808 +/- .009** | not headline | high but not central |
| Sentence LM | 79.6 +/- 1.3 | .777 +/- .013 | — | — |
| MLP JEPA | 198.7 +/- 0.5 | .406 +/- .008 | .346 | .895 |
| Token LM | 3.5 +/- 0.8 | .533 +/- .048 | — | — |
| Causal JEPA | **215.3 +/- 2.2** | .403 +/- .007 | .332 | .846 |

JEPA is not collapsed and strongly exposes progress-correlated information. High effective rank does not explain planning because the low-rank token model is competitive and the highest-rank causal JEPA plans worse than the MLP JEPA. Sentence-model features make operation identity much more linearly accessible.

The necessary-action probe is not positive evidence: its balanced accuracy is exactly .500 despite raw accuracy around .878, which comes from label imbalance. Resolved count is confounded with elapsed step index. Numerical outcomes are weakly decoded, with R-squared around .09 for MLP JEPA, .05 for causal JEPA, and approximately zero for the language models.

PCA coordinates and preliminary figures exist, but there is no validated cross-model t-SNE or UMAP result showing paraphrase clustering and negation separation. Controlled minimal pairs, full-space retrieval, action-paraphrase invariance, operator swaps, irrelevant renaming, and progress-versus-step deconfounding remain planned rather than completed.

### 14. Length generalization

The first exact-length evaluation accidentally reduced distractors as length increased and is excluded. A corrected one-seed evaluation keeps three irrelevant variables at every length. For sentence LM, strict success falls from .27 at length 9 to .10 and .11 at lengths 10 and 11. Sentence-plus-latent falls from .21 to .16 and .12. Slack-two falls from .77 to .67/.63 and from .74 to .64/.60, respectively. Slack-four saturates at 1.0 because only three distractors exist and is uninformative. Matched JEPA and token curves, more seeds, and harder non-ceiling budgets are still missing.

### 15. Process failures and repairs

The project encountered many failed, timed-out, and partially submitted rounds. The main scientific repairs were:

| Failure | Repair | Status |
|---|---|---|
| Context-window jobs exceeded time caps | Larger caps and smaller diagnostics | Recovered |
| Counterfactual targets omitted causal prefixes | Full-state causal-prefix construction and tests | Repaired locally; did not solve ALFWorld |
| EMA/dropout inconsistency risk | EMA always evaluation mode; dropout zero | Current invariant |
| Counterfactuals lacked latent prediction in old recipes | Added explicit counterfactual next-state loss | Supported gain |
| Deep search summed edge Energies | Terminal endpoint only | Repaired |
| Beam search was shooting/global pruning | Root-balanced genuine beam | Repaired |
| Early terminal path length leaked success | Absorbing padding with requested horizon | Repaired |
| Dense Energy compared different horizons | Within-horizon comparisons | Repaired; still negative |
| Slurm archive/import failures | Snapshot/archive fixes and reruns | Recovered for cited rows |
| GAR pairwise NaNs | Finite-regression/NaN repair | Recovered |
| Full catalogue treated invalid actions as neutral no-ops | Absorbing failure-state variant | Tested; insufficient |

Failed runs remain in controller history and are not silently counted as negative scientific results. Several old status documents describe superseded GAR recipes; the method handbook and this report are the most recent synthesis, with the caveat below about horizon semantics.

## The intuitive picture

![A flow diagram shows the current Endpoint-Energy JEPA and four evidence boxes. The model encodes observed history, recursively predicts candidate endpoints, ranks terminal endpoints, executes one real action, and replans. Evidence boxes show strong candidate-privileged depth gains, failure without action menus, failure of dense endpoint supervision, and the unresolved horizon-support mismatch.](figures/current_method_and_evidence.svg)

The figure separates the successful mechanism from its present boundary. Recursive endpoint ranking works extremely well when future feasible actions are supplied, but unrestricted catalogue selection fails and the Energy semantics across all evaluated depths still need one clean repair.

## The technical details

### Current endpoint model

```text
h_t    = prompt plus observed outcome history
z_t    = online encoder(h_t)
u(a)   = learned 16-dimensional action embedding
z_hat1 = F(z_t, u(a_1))
z_hat2 = F(z_hat1, u(a_2))
...
z_hatH = F(z_hat(H-1), u(a_H))
E_H    = Head(z_t, z_hatH, z_0, H)
```

The state width is 256. A token Transformer encodes sentences and actions; a discourse encoder produces the history state. The validated predictor is a residual concatenation MLP reused at every imagined step. The Energy head receives the root state, predicted endpoint, initial problem state, and a scalar horizon encoding `log(1+H)/4`. Lower Energy is better.

EMA encoders produce stopped-gradient target endpoints and solved-state targets. They remain frozen and in evaluation mode. All dropout is zero.

### Endpoint training target

For each fresh problem:

1. Generate a valid solution trajectory.
2. Sample one anchor state.
3. Select the factual next root and two alternative feasible roots.
4. Sample continuations beneath each root.
5. Execute each sequence in the environment to obtain its true endpoint.
6. Encode the true endpoint and solved trajectory with EMA modules.
7. Recursively imagine the same action sequence with the online predictor.
8. Order the imagined endpoints by the normalized L1 distance of their true EMA endpoints to the EMA solved state.
9. Apply logistic pairwise ranking so better endpoints receive lower Energy.

The solved-state representation constructs training labels only. It is unavailable to the deployed planner. Counterfactual roots receive true one-step latent targets. The current best fixed-H4 model backpropagates the final Energy through all four predictor applications but does not apply dense intermediate latent or Energy losses.

### Current planner

At each real step, root-balanced beam search keeps a separate continuation budget, usually eight beams, beneath every first action. Candidate sequences are recursively imagined. Each final endpoint is scored once:

```text
sequence score = E_D(z_t, F repeated D times, z_0, D)
```

No edge scores are summed. The first action of the lowest-Energy final beam is executed. The real environment returns an outcome sentence, the model re-encodes the observed history, and planning restarts.

If a candidate solves early, its state becomes absorbing, but the requested depth remains fixed. This removes terminal-length leakage.

### The present method inconsistency

The handbook currently calls the mixed-horizon model the paper method, while recent simplification work favors fixed H4. Neither is yet a completely coherent test-time-scaling method:

- Fixed H4 has one clean endpoint meaning at D4, but D1, D8, and D16 evaluate the horizon-conditioned head away from its training horizon.
- Mixed {1,2,4,8} trains several endpoint horizons, but D3 partial pruning and D16 terminal scoring are out of support.
- Root H4-to-H1 distillation teaches the same Energy head to represent downstream H4 root quality at H1 while H4 represents endpoint quality. This improves transfer but mixes semantics.

The clean next formulation should train one final-endpoint Energy with identical semantics at every supported horizon, sample H from {1,2,4,8,16}, include every reported/pruned depth or avoid scoring unsupported partial depths, and remove root distillation from that Energy. If a strong one-step policy is desired, it should be a separate named head rather than altering endpoint-Energy semantics.

### Reproducibility pointers

- Stable project entry: [`projects/intent_phrase/README.md`](../../../../projects/intent_phrase/README.md)
- Method handbook: [`projects/intent_phrase/method/README.md`](../../../../projects/intent_phrase/method/README.md)
- Endpoint training: [`scripts/train_intent_horizon_energy.py`](../../../../scripts/train_intent_horizon_energy.py)
- Endpoint evaluation: [`scripts/evaluate_intent_terminal_energy.py`](../../../../scripts/evaluate_intent_terminal_energy.py)
- Five-seed terminal-safe round: [`2026-08-05-intent-terminal-safe-multihorizon-training-v1`](../../../../runs/autonomy/intent_phrase/2026-08-05-intent-terminal-safe-multihorizon-training-v1)
- Recent simplification round: [`2026-08-06-intent-simple-endpoint-method-ablation-v3`](../../../../runs/autonomy/intent_phrase/2026-08-06-intent-simple-endpoint-method-ablation-v3)
- Full-catalogue and dense round: [`2026-08-06-intent-invalid-semantics-and-dense-endpoints-v1`](../../../../runs/autonomy/intent_phrase/2026-08-06-intent-invalid-semantics-and-dense-endpoints-v1)
- Dense repair round: [`2026-08-06-intent-within-horizon-dense-endpoint-repair-v1`](../../../../runs/autonomy/intent_phrase/2026-08-06-intent-within-horizon-dense-endpoint-repair-v1)
- Earlier comprehensive representation report: [`2026-08-03-project-state-and-representation-analysis`](../2026-08-03-project-state-and-representation-analysis/REPORT.md)
- Current ICLR draft: [`projects/intent_phrase/paper/main.pdf`](../../../../projects/intent_phrase/paper/main.pdf)

### Live controller state

At the refresh used for this report, all 15 jobs in `2026-08-06-intent-simple-endpoint-method-ablation-v3` were complete. The H1 fixed-teacher duplicate/remainder on Alex, job 3958368, was still marked running in the partially submitted v2 round. The controller is paused, so no autonomous follow-up submission will occur. Other active jobs shown by the global controller belong to other TextJEPA subprojects and are outside this report.

## What we can conclude

### Direct observations

- JEPA can perform nontrivial closed-loop selection among language intent phrases.
- Under a shared feasible-action interface, the five-seed MLP GAR-JEPA is competitive with token and sentence language models, though sentence-plus-latent is better in strict mean.
- Latent prediction alone is not enough for good planning in this environment.
- Pairwise ranking supervision is the largest repeatedly observed mechanism gain.
- Counterfactual transition supervision adds useful information beyond endpoint ranking.
- A recursively predicted final endpoint can be ranked well enough to produce large depth gains under a symbolic future feasible-action tree.
- Terminal-safe five-seed depth results are stable, but the horizon-support mismatch prevents a final clean scaling claim.
- Fixed H4 is the strongest observed training horizon; longer fixed horizons are worse.
- Dense intermediate endpoint ranking is strongly harmful in the tested formulations.
- More than two sampled continuations per root is unnecessary in the current one-seed screen.
- Full-catalogue action selection fails, even with absorbing invalid states and balanced invalid sampling.
- Hierarchy has not improved the intent-phrase method after correcting confounds.
- JEPA representations are high-rank and expose progress-correlated quantities, while sentence-model states expose operation identity more linearly.

### Supported interpretation

Prediction supplies a useful latent transition model, but planning success depends on learning local or endpoint action ordering. The current evidence favors the paper thesis “prediction alone is not planning” over claims that JEPA universally outperforms language models. Endpoint ranking is a simpler and stronger deep-search mechanism than the many calibrated advantage/state-value variants tested so far.

The method presently solves a controlled candidate-valuation problem. That is scientifically useful because it isolates whether learned latent consequences can support planning. It is not equivalent to solving unrestricted language reasoning, generating actions, or inferring feasibility.

## What we cannot conclude

- We cannot yet claim clean monotonic test-time scaling. The mixed-horizon curve is terminal-safe but evaluates some unsupported horizons.
- We cannot claim unrestricted planning without symbolic action support; the full-catalogue gate fails.
- We cannot claim JEPA is better than generative models. The sentence-plus-latent baseline leads strict success.
- We cannot claim predictive factorization is better than direct ranking. The one-seed direct ranker matches local GAR in-distribution.
- We cannot claim broad language reasoning from stylized iGSM.
- We cannot claim length generalization; current evidence shows degradation and is incomplete across families.
- We cannot claim that GAR or endpoint ranking creates a globally meaningful Euclidean geometry. Learned heads outperform raw distance.
- We cannot claim that high effective rank, linear probe accuracy, or t-SNE/UMAP appearance explains planning.
- We cannot claim paraphrase invariance, negation sensitivity, causal-role clustering, or successful sentence reconstruction because those analyses are not complete.
- We cannot claim exact parameter- or FLOP-matched superiority.
- We cannot claim the endpoint simplification ablations are final because most use one seed.

## What happens next

### Decision 1: make endpoint Energy scientifically coherent

This is the highest-value immediate experiment.

| Cell | Training horizons | Root distillation | Dense Energy | Purpose |
|---|---|---:|---:|---|
| Clean supported Energy | {1,2,4,8,16} | 0 | 0 | Same endpoint semantics at every reported horizon |
| No horizon input | {1,2,4,8,16} | 0 | 0 | Test whether one shared Energy is sufficient |
| Fixed H4 reference | {4} | 0 | 0 | Clean single-horizon upper/reference at D4 |
| Current mixed reference | {1,2,4,8} | .25 | 0 | Quantify the semantic-confound benefit |

Evaluation should include depths 1, 2, 4, 8, and 16. Beam pruning must either score only supported depths or retain root-balanced candidates without an unsupported intermediate Energy call. The decision criterion is replicated increasing or saturating success with depth, not merely a high D4 number.

### Decision 2: isolate recursive dynamics supervision

After the Energy semantics are fixed, compare one change at a time:

| Dynamics cell | Endpoint ranking | Intermediate latent MSE | Endpoint latent MSE | Gradient through rollout |
|---|---:|---:|---:|---:|
| Recursive endpoint only | yes | no | no | yes |
| Final latent target | yes | no | yes | yes |
| Dense latent targets | yes | yes | yes | yes |
| Detached rollout | yes | no | no | no through earlier predictions |

Measure rollout error, endpoint rank, gradient cosine/conflict, off-manifold distance, and closed-loop success. This will distinguish optimization conflict from simple predictive drift.

### Decision 3: establish mechanism controls

- Confirm the selected endpoint model over five seeds.
- Train a matched direct sequence ranker that sees the same candidate sequences and labels but cannot use recursively predicted states.
- Compare exact encoded endpoints, predicted endpoints, shuffled labels, and oracle endpoint distances.
- Correlate pairwise rank accuracy, top-1 root recall, regret, margin, and rollout drift with strict success across seeds and checkpoints.
- Test data efficiency and longer compositions, where predictive factorization has a plausible advantage over direct discrimination.

### Decision 4: finish representation science

Construct controlled pairs for paraphrase, negation, operator swap, irrelevant entity renaming, matched elapsed step with different remaining work, and action paraphrase. Report full-space retrieval and ranking invariance first. Use PCA, t-SNE, and UMAP only as illustrations. Compare scalar endpoint Energy, raw goal distance, full latent state, and the residual after removing a learned progress direction.

### Decision 5: paper breadth

Once the coherent iGSM mechanism is frozen, add one small non-arithmetic language-action domain before attempting the entire four-domain suite. Textual BlocksWorld, relational graph navigation, or a logic-dependency planner is lower risk than ALFWorld. Faithful iGSM should still receive the frozen recipe and matched token/sentence baselines. ALFWorld should remain excluded until candidate coverage and causal counterfactual supervision pass admission gates.

### Decision 6: final evaluation package

- Five seeds for every headline row and central ablation.
- At least 1,000 paired test puzzles per seed.
- A single first-solution evaluation yielding slack 0, 1, 2, 3, 4, and area-under-budget curves.
- Held-out lengths with matched distractor counts and no slack ceiling.
- Correct measured FLOPs and wall time for looped LMs and JEPA search.
- Paired bootstrap confidence intervals for model differences.
- Explicit interface labels in every table and figure.

## Words used in this report

- **Action catalogue:** All action phrases associated with a problem, including actions that are not currently executable.
- **Candidate-privileged:** Symbolic or oracle information supplies candidate actions that an unrestricted learned agent would have to infer.
- **Counterfactual:** An alternative action or action sequence not taken on the reference solution trajectory.
- **Dense supervision:** A loss applied at intermediate rollout steps rather than only at the final endpoint.
- **EMA:** Exponential moving average; a slowly updated target copy of the model that does not receive gradients.
- **Endpoint Energy:** A learned lower-is-better scalar assigned to a recursively imagined final state.
- **GAR:** Geometric Advantage Ranking; the earlier local pairwise action-ordering mechanism.
- **iGSM:** A synthetic arithmetic dependency-graph problem generator.
- **JEPA:** Joint-Embedding Predictive Architecture; a model trained to predict representations rather than reconstruct all text.
- **Latent state:** A vector representation of the prompt and observed interaction history.
- **Root-balanced beam:** A search that reserves continuation capacity separately under each possible first action.
- **Slack:** The number of extra actions permitted beyond the shortest solution.
- **Strict success:** Solving with zero excess actions.
- **Terminal-safe:** Early success cannot be detected from candidate length or horizon metadata.
- **Test depth:** The number of future actions recursively imagined before executing the first action.
- **Training horizon:** The number of actions used to construct one endpoint-ranking training example.

## Questions for you

- Should the coherent multi-horizon Energy experiment take priority over five-seed replication of the fixed-H4 D4 result?
- Should the paper present the symbolic feasible future tree as the primary controlled laboratory interface, or require a learned proposal component before treating test-time scaling as headline evidence?
- After iGSM, should the first breadth domain be textual BlocksWorld, graph navigation, or a compact logic planner?
