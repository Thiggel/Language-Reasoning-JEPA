# ALFWorld is executable, but paper training still awaits admission gates

## The one-sentence answer

The real ALFWorld engine now collects and exactly replays non-oracle intent trajectories, and all four model process pathways have passed after two queue-delayed paths were recovered on Grünau.

## First, the idea in everyday language

Imagine teaching someone to cook by letting them read a room description, choose an instruction such as “take the apple from the counter,” and observe what changes. A fair test cannot whisper the list of legal moves before every choice. We therefore build possible instructions only from objects the learner has already seen, while using ALFWorld’s hidden expert and legal-move list only to prepare labels and verify correctness. The implementation must also reopen the same simulated room and reproduce every observation exactly; otherwise apparent reasoning gains could merely reflect inconsistent data.

## Why this question matters

ALFWorld is the paper campaign’s interactive, non-arithmetic domain. It tests whether consequence geometry transfers beyond synthetic equations. A broken collector, oracle action menu, or drifting replay would invalidate an entire learning-rate sweep. This report supports a narrow decision: run bounded data-admission gates and recover the two scheduler-delayed model gates, but do not start ALFWorld model selection yet.

## What we tested

We installed the official text-only ALFWorld 0.5.0 engine from a pinned source revision and downloaded its official game files. One training, one seen-validation, and one unseen-test game were collected with the hand-coded expert. Every factual step received one recoverable alternative action with up to four teacher continuation observations. Collection used the privileged legal-action list only for labels. Interactive replay regenerated candidate actions from the initial room description and accumulated observation history.

Separately, the four model process gates asked only whether each implementation could train for two epochs, reload a checkpoint, and emit finite closed-loop metrics. Geometry JEPA and recurrent sentence prediction completed in the original recovery round. Recurrent token prediction and recurrent sentence-latent prediction remained pending externally, then completed in duplicate recovery jobs on free Grünau GPUs.

## What a fair comparison means here

At evaluation, the model sees natural-language history, the goal, and a fixed-grammar catalogue grounded in observed entity names. It never receives ALFWorld’s current legal-action menu, expert plan, future availability, or teacher continuation. The generated catalogue intentionally contains invalid cross-products, so feasibility must be learned. Candidate ordering is a stable hash that also covers objects discovered later. Training may use privileged feasibility and continuation labels only in explicitly labelled objectives. All paper methods must receive the same catalogue and evaluation budgets.

## What happened

| Check | Sample | Outcome | Interpretation |
|---|---:|---:|---|
| Official expert collection | 3 episodes, 36 steps | 100% completed | All train/seen/unseen smoke episodes were solvable |
| Factual deterministic replay | 3 episodes, 36 steps | 100% exact | Stored observations and terminal success reproduced |
| Recoverable alternatives | 36 alternatives | 36 recovered | One non-expert branch per factual step reached the goal |
| Dynamic catalogue replay | 1 unseen episode, 16 steps | 100% expert recall | No legal menu was used by the evaluator |
| Geometry JEPA process gate | 1 run | completed | Training, checkpoint reload, and metrics path work |
| Recurrent sentence model process gate | 1 run | completed | Training, checkpoint reload, and metrics path work |
| Recurrent token / sentence-latent gates | 2 recovery runs | completed | Training, checkpoint reload, and finite metrics paths work |
| Unseen ALFWorld schema recovery | 4 episodes, 67 steps | completed | Raw oracle menus compile to feasibility only within the non-oracle catalogue; exact replay remains 100% |

These are engineering-validity observations, not estimates of model quality. The completed two-epoch models solved zero validation episodes, which is unsurprising at this scale and is not used to rank methods.

## The intuitive picture

![Flow showing observed text generating a broad action catalogue, an action entering the ALFWorld engine, and the resulting observation returning to history while the oracle menu remains label-only](alfworld-information-boundary.svg)

The figure separates deployment information from privileged collection labels. The loop on the left is what a model may use; the dashed label-only path must never enter policy inference.

## The technical details

The catalogue generator parses numbered entities only after phrases such as “you see,” treats reset-time entities as navigable receptacles, and grounds a fixed grammar for navigation, opening, taking, moving, heating, cooling, cleaning, slicing, examining, inventory, and lamp use. It deliberately emits invalid object–receptacle cross-products. Collection runs the official hand-coded expert, checks that every expert action occurs in both the generated catalogue and privileged admissible set, branches from the exact factual prefix, and follows the expert after each alternative to retain only recoverable counterfactuals.

Fast Downward maps a private planner library for each engine and does not release the deleted mapping until process exit. Repeated environments therefore exhausted temporary storage despite normal close calls. Collection now resets one engine for all branches in an episode and executes each episode in a disposable worker. Interactive evaluation likewise owns each engine in a spawned subprocess and explicitly closes it. This was directly verified: five in-process engines accumulated fifteen deleted planner mappings before the repair, whereas the process-isolated validator completed exact replay without retaining them in the model process. The padded episode-level action axis also carries a per-step observed-candidate mask, so actions grounded from later-discovered objects receive no earlier support loss. Fifteen focused tests pass, including this future-entity regression, the zero-counterfactual regression, schema validation, non-oracle catalogue construction, and closed-loop planning.

## What we can conclude

We directly observe that the adapter can collect and replay official train, seen-validation, and unseen-test games without exposing the oracle menu during interactive evaluation. We also observe that all four core model pathways have passed their process gates. The first four-game unseen gate exposed and then passed a domain-neutral schema repair: the full privileged legal menu stays raw, while compiled feasibility is defined only over candidates the model actually receives.

## What we cannot conclude

We cannot yet claim ALFWorld dataset admission, because split-scale identity checks, tiny-set overfitting, random and oracle bounds, action-shuffle degradation, and model closed-loop success remain untested. We cannot compare model quality from two-epoch zero-success process runs. The smoke and four-game unseen samples are too small to estimate recovery coverage. The external Alex and Grete duplicates remain scheduler-pending, but they are no longer needed for the process decision.

## What happens next

Run independent train, seen-validation, and unseen-test collection gates with exact interactive replay. In parallel, repeat only the two pending model pathways on currently free Grünau GPUs. If all five jobs complete, merge a larger deterministic dataset, verify disjoint identities, run tiny-set overfit and random/oracle/action-shuffle controls, and only then admit the first 27-run learning-rate cell. Any catalogue miss, replay drift, temporary-storage growth, or missing artifact blocks ALFWorld scaling.

## Words used in this report

- **ALFWorld:** A text-based household simulator with goals such as moving or preparing objects.
- **JEPA:** Joint-embedding predictive architecture, a model that predicts consequences in a learned latent space.
- **Oracle menu:** The simulator’s privileged list of actions that are legal in the current hidden state.
- **Counterfactual:** A recorded alternative action and the consequence that would have followed it.
- **Process gate:** A small run checking that software trains, reloads, and evaluates before expensive experiments.

## Questions for you

- After these gates pass, should the next priority be the full ALFWorld dataset admission battery or the first complete iGSM learning-rate cell?
- Is one recoverable alternative per ALFWorld step sufficient for the first model pilot, or should collection spend roughly twice the time for two alternatives immediately?
