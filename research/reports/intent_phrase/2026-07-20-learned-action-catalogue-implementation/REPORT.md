# A non-oracle action catalogue for intent-phrase planning

## The one-sentence answer

The intent-phrase planner can now simulate multiple steps using only a fixed prompt-derived action catalogue and learned availability and policy heads, with symbolic feasibility excluded from proposal construction and 119 repository tests passing; whether this improves task accuracy is deliberately left to the submitted pilot.

## First, the idea in everyday language

Imagine planning errands in an unfamiliar building. A privileged planner is handed a fresh list of doors that can be opened after every imagined move. A deployable planner instead starts with the building directory, predicts which doors should be open from its imagined location, and considers only the most plausible ones. The old depth-two and depth-four intent-phrase planner behaved like the privileged planner. The new implementation behaves like the second.

Every arithmetic problem supplies a finite directory: one outcome-free intent phrase for every variable described in the prompt. A learned availability head estimates whether an action can execute in an observed or imagined reasoning state. A separate behavioral prior estimates whether an action resembles the demonstrated useful policy. The joint-embedding predictive architecture (JEPA) then predicts the consequences of the proposed action sequences and chooses which first action to execute.

## Why this question matters

The existing depth-four result improved strict length-nine success from .061 to .439, but it enumerated future legal actions from the hidden reference graph. That is useful as an upper bound, not as a deployable result. This implementation removes that hidden future-action channel. If accuracy now improves with simulation depth relative to a compute-matched prior-only beam, the project can support the paper-relevant claim that learned latent simulation adds value beyond proposal generation.

## What we tested

This cycle tested implementation validity rather than scientific performance.

The new planner was exercised at depths one through four. Tests cover non-oracle multistep construction, absence of feasibility queries inside proposal generation, one beam per root action, global expansion limits, invalid execution, checkpoint eligibility, predicted-state availability supervision, predicted-state prior supervision, and gradient isolation. A real 71.7 MB action-prior checkpoint was loaded for an end-to-end CPU smoke. A tiny head-only training run initialized all 354 checkpoint tensors, trained 0.28 million head parameters, and emitted finite losses and metrics.

The final full repository suite contains 119 passing tests. The first full-suite invocation failed before collection because the worktree root was absent from `PYTHONPATH`; rerunning with both the root and `src` matched the repository import layout and passed. This setup error is not counted as a model or planner failure.

## What a fair comparison means here

The prior-only control and JEPA planner receive exactly the same prompt catalogue, learned action scores, top-four roots, root-balanced future beams, and maximum expansion count. They differ only in final trajectory scoring:

- Prior-only chooses the trajectory with the best accumulated learned proposal score.
- JEPA chooses among the identical trajectories using predicted latent consequences and its learned remaining-cost energy.

No current or future feasible-action menu may affect proposal scores or filtering. Ground-truth feasibility and query ancestry are read only after proposals have been produced, to calculate audit metrics. An invalid selected action terminates that evaluation episode as a failure; the evaluator never substitutes a legal action.

The first pilot freezes the encoder, action encoder, causal predictor, value head, and exponential-moving-average target encoders. Only the availability and behavioral-prior heads train. This prevents an availability auxiliary from silently changing JEPA dynamics and makes the comparison interpretable.

## What happened

| Validation item | Expected behavior | Result |
|---|---|---|
| Non-oracle depth > 1 | Constructor permits learned catalogue without oracle flag | Passed |
| Feasibility isolation | Proposal generation makes zero symbolic feasibility calls | Passed |
| Root balance | Every selected root retains its own bounded future beam | Passed |
| Expansion cap | Total candidates never exceed `max_expand` | Passed |
| Invalid action | Counted and episode fails; no fallback | Passed |
| Predicted-state support | True, one-step, and recursive states are supervised | Passed |
| Predicted-state prior | Same three state streams train the policy head | Passed |
| Gradient boundary | Head losses do not update state/action encoders or predictor | Passed |
| Legacy behavior | Existing feasible-menu planner tests remain valid | Passed |
| Full regression suite | All repository tests | **119 passed** |
| Tiny checkpoint training | Intended tensors load and only heads train | Passed; 0.28M trainable parameters |

No scientific success rate from the untrained-support smoke is interpreted. Its invalid-action rate was intentionally exposed by the new metrics and motivated a hard checkpoint-validity gate.

## The intuitive picture

![A fixed prompt-derived catalogue flows into learned availability and behavioral-prior heads. Their top actions are expanded independently per root using predicted states. The same trajectory bank is sent either to accumulated prior scoring or JEPA consequence scoring, while the hidden symbolic environment is used only after selection for real execution and audit metrics.](learned_catalogue_flow.svg)

The boundary to notice is the dashed line: hidden feasibility is outside the proposal loop. Both competing decision rules see the same learned trajectory bank.

## The technical details

### Information boundary

At episode start, the catalogue contains every action phrase rendered from the definitions already present in the prompt. It contains no outcome sentence, current availability bit, relevance label, remaining-step label, or future menu. Executed actions are removed using the planner's own action history. Catalogue order is shuffled deterministically per problem/seed so equal scores cannot recreate the known variable-order artifact.

The proposal function does not access `SymbolicEnv.feasible_actions`, the hidden dependency traversal helper, query ancestry, or terminal detection. A unit test patches symbolic feasibility to raise immediately and still constructs depth-three trajectories successfully.

### Two-head proposal model

For predicted state \(\hat s\) and action phrase embedding \(a\), the availability head emits \(f(\hat s,a)\), while the behavioral prior emits \(\pi(\hat s,a)\). The proposal log-score is

\[
S(\hat s,a)=w_\pi\log\operatorname{softmax}_a\pi(\hat s,a)
+w_f\log\sigma(f(\hat s,a)).
\]

The pilot fixes \(w_\pi=1\) and evaluates \(w_f\in\{0.5,1,2\}\). This sweep matters because cross-entropy prior logits and binary availability logits do not have a universal common scale.

The prior remains trained only to distinguish the demonstrated action among truly feasible actions. The availability head alone learns feasible versus infeasible. This separation makes failure diagnosis possible and avoids pretending that behavioral cloning is action-free JEPA supervision.

### Root-balanced learned beam

At a real state, the planner scores the complete unexecuted catalogue and retains the top \(M\) roots. For each root separately, it recursively:

1. predicts the latent endpoint of the current candidate sequence using the causal predictor and full observed history;
2. scores every not-yet-used catalogue action at that predicted state;
3. expands the parent with its top \(M\) actions;
4. retains at most \(B\) lowest-cost descendants for that root.

The effective per-root beam is additionally capped by `floor(max_expand / number_of_roots)`. An error is raised if `max_expand` cannot retain at least one sequence for every selected root. Consequently, one root cannot win merely because it generated more descendants.

The prior-only controller minimizes accumulated negative proposal score. The JEPA controller ignores proposal multiplicity and reranks the identical final sequence list using its latent rollout and value energy. Receding-horizon execution takes only the first action, observes the real rendered outcome if valid, re-encodes the state, and replans.

### Invalid actions and audit metrics

The environment is queried only when the selected root is actually executed. If it is illegal, `env.step` raises, the episode records one invalid action, and evaluation returns failure immediately. There is no feasibility retry, first-valid fallback, or deletion of the bad root.

Reported learned-catalogue diagnostics are:

- invalid selected-action rate;
- recall of truly feasible roots among the proposed roots;
- precision of proposed roots that are truly feasible;
- recall of currently feasible query-relevant roots;
- ordinary strict and excess-action success.

Ground-truth feasibility and relevance used in these metrics are post-hoc evaluator labels and never feed the model.

### Training on imagined states

Both heads see three aligned state streams at every factual training position:

1. the true encoded prefix state;
2. the one-step teacher-forced JEPA prediction;
3. the recursively predicted open-loop state.

Targets are shifted so each predicted prefix is paired with the availability menu and demonstrated action appropriate to that same next decision. Both state and action inputs are detached. The first pilot therefore changes only the two heads. Existing checkpoints lacking the new configuration fields retain historical behavior through backward-compatible defaults.

### Checkpoint gate

Evaluation refuses to label a run learned-catalogue planning unless its stored configuration proves all of the following:

- the action prior exists;
- all-action supervision was enabled;
- action-feasibility loss had positive weight;
- action-prior loss had positive weight;
- availability was trained on all three state streams;
- the prior was trained on all three state streams.

This prevents a technically runnable but scientifically invalid random-support checkpoint from entering a table.

### Files to review

- Planner and validity gate: `src/textjepa/planning/search.py`
- New audit metrics: `src/textjepa/planning/evaluate.py`
- Predicted-state head construction: `src/textjepa/models/discourse_jepa.py`
- Multi-state prior loss and metrics: `src/textjepa/objectives/macro_hierarchy.py` and `src/textjepa/training/trainer.py`
- Head-only training recipe: `configs/experiment/paper_causal_j3_learned_catalogue_heads.yaml`
- Reproducible train/evaluate wrapper: `scripts/run_intent_learned_catalogue_head_pilot.sh`
- Adversarial tests: `tests/test_planning.py` and `tests/test_model.py`

## What we can conclude

Direct observation: the code now has a multistep action-proposal route whose trajectory construction does not query symbolic feasibility. Direct observation: the head-only gradient boundary, tensor alignment, invalid-action semantics, checkpoint gate, expansion cap, and old planner behavior pass targeted tests and the full suite.

Supported engineering inference: the implementation is safe enough for a bounded one-seed, two-learning-rate validity pilot. It is not yet safe to launch a dense planning-depth, proposal-width, GAR-beam, model-width, and seed campaign simultaneously.

## What we cannot conclude

We cannot yet conclude that learned availability has adequate recall, that the behavioral prior is calibrated on infeasible candidates, that JEPA beats prior-only search, or that accuracy improves with depth. The proposal prior was trained on feasible-menu behavior cloning, so its logits on infeasible actions are unconstrained; the availability weight sweep is necessary. Training on factual predicted trajectories does not fully reproduce counterfactual beam-state drift. If support fails mainly on counterfactual states, counterfactual support supervision or dataset aggregation may be required.

GAR still uses true environment continuations while constructing training targets. Removing oracle menus from evaluation does not make GAR interaction-free. GAR beam width is intentionally held at the historical value in this first pilot; a matched \(B=1/4/8\) training comparison follows only if learned proposal validity passes.

## What happens next

The pilot adapts the two heads from the existing seed-zero action-prior checkpoint at learning rates \(3\times10^{-4}\) and \(10^{-3}\). It evaluates 30–60 fixed problems at lengths six and nine, strict and plus-two budgets, depths one, two, and four, and availability weights 0.5, 1, and 2. Prior-only and JEPA scoring share every proposal.

The mechanism advances only if:

- invalid selected-action rate is low enough to leave a meaningful success sample;
- top-four feasible and necessary-action recall are healthy;
- increasing depth does not merely increase invalid trajectories;
- at some matched setting, JEPA improves strict success by at least .05 over prior-only or shows a clear depth trend worth confirming.

If proposal validity is poor, tune availability/proposal construction before GAR. If proposals are healthy but JEPA remains worse, diagnose value calibration and rollout drift. If the pilot works, repeat the chosen learning rate and availability weight on the other two existing seeds, then test GAR teacher beams 1, 4, and 8.

## Words used in this report

- **Action catalogue:** The fixed set of outcome-free intent phrases derived from a problem prompt.
- **Action prior:** A supervised head that predicts which feasible action the demonstrated policy would choose.
- **Availability head:** A binary head predicting whether an action can legally execute in a state.
- **Beam:** A bounded collection of partial action sequences retained during search.
- **Candidate-privileged:** A protocol receiving action candidates from hidden environment state.
- **JEPA:** Joint-embedding predictive architecture, a model predicting latent representations rather than reconstructing text.
- **Latent state:** A learned vector summarizing the reasoning history.
- **Oracle menu:** The exact actions known by the hidden environment to be feasible at an imagined state.
- **Root-balanced:** Reserving the same maximum number of future sequences for every first action.

## Questions for you

- Should an invalid selected action remain an immediate episode failure, as implemented, or consume one action and expose an explicit rejection observation to the policy?
- After the pilot, should the first scale-up prioritize broader proposal width or the matched GAR beam-width comparison?

