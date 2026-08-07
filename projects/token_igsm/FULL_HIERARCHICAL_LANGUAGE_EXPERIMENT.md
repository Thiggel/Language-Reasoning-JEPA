# First full strict-nested language-planning experiment

This experiment is the first executable test of the complete revised
hierarchy. It is distinct from the earlier flat, candidate-privileged token
oracle runs.

## Shared trained hierarchy

The representation and dynamics path is strictly nested:

\[
E_{\mathrm{LM}}(x_{<n})=h_n,
\qquad z_n^0=E_0(h_n),
\qquad z_j^1=E_{0\to1}(z_{B_j}^0).
\]

`P0` predicts token transitions with its complete bounded token history.
`A1` encodes complete newline/EOS reasoning actions, and `P1` predicts
sentence transitions with its complete bounded sentence history. `Pi1` sees
only the pre-action sentence state/context and the fixed task embedding
`H(prompt_len)`. Counterfactual reasoning steps train both dynamics levels;
only symbolically verified complete steps enter sentence-level evaluation.
The shared task projection is trained with `Pi1` at the macro stage and then
frozen during value-only distillation, preventing the value learner from
shifting the frozen macro prior's conditioning distribution.

The high-level manager samples supported macro actions in `Pi1` coordinates.
Its first predicted successor is the sentence waypoint. The primary token
worker is no longer a one-shot Qwen sample-and-rerank procedure. At every live
prefix, Qwen supplies only supported next-token proposals; `P0` predicts their
consequences and JEPA waypoint discrepancy selects the surviving beam or CEM
elites. Qwen log likelihood is absent from the primary worker objective. A
small likelihood penalty and likelihood-only selection are isolated
ablations, while one-shot reranking is retained as the historical control.

The worker-search comparison contains: full-prefix JEPA beam search,
elite-prefix autoregressive CEM, sparse first-order Markov CEM, and an
independent-position categorical CEM negative control. The autoregressive
methods preserve complete prefixes; they never splice independently selected
future token positions.

Two hierarchy execution policies are evaluated. Open-loop execution retains
the complete high-level waypoint sequence while the worker realizes it.
Closed-loop execution commits only to the first waypoint, exactly re-encodes
the achieved worker state, replaces the imagined manager state, and reruns
high-level CEM. After every executed reasoning step, the frozen LM and both
bounded predictor histories are rebuilt from the real prefix.

High-level support is separately ablated as ambient-action CEM, `Pi1`-coordinate
CEM, `Pi1`-coordinate CEM with a noise trust region, and the latter plus an
explicit prior NLL cost. This distinguishes proposal support from an additive
likelihood objective.

## Two evaluation modes

- **Oracle-terminal manager (candidate privileged):** the successful terminal
  sentence embedding set is exposed to high-level CEM. This diagnoses the
  representation, dynamics, macro prior, and worker interface; it is not a
  deployable accuracy claim.
- **Distilled-value manager (no terminal input):** the terminal set is used
  offline to create first-action rankings. At evaluation, the manager receives
  only the current causal sentence state and task embedding. The worker and
  exact re-encoding path are identical to oracle evaluation.

CEM elites and random controls are grounded through the token worker. Grounded
costs—not merely model-predicted costs—select subsequent CEM elites. Every
branch retains its own bounded `P1` state/action history.

Offline value-teacher trajectories obey the same contract. Every requested
macro action is realized as a complete Qwen sentence, rolled through `P0`,
exactly re-encoded, converted to `A1(text)`, and rescored under `Pi1` at the
original pre-action state. Verified terminal sets and achieved rollout
validity are stored separately. Latent-only `P1` rollouts are diagnostic
artifacts and are rejected by value training.

## Isolated 2x2 value ablation

One shared token/sentence/macro checkpoint feeds four value students:

| Local endpoint discrepancy | Teacher closure | Meaning |
| --- | --- | --- |
| Euclidean | one-step raw endpoint | neither addition |
| Mahalanobis | one-step raw endpoint | Mahalanobis only |
| Euclidean | supported multi-depth search | quasimetric only |
| Mahalanobis | supported multi-depth search | full method |

The raw teacher uses terminal discrepancy only. The quasimetric teacher uses
total prefixes 1--8 plus step and macro-prior costs. Thus “quasimetric” means
the prior-constrained Bellman/search closure of the local endpoint discrepancy,
not Mahalanobis distance itself.

Euclidean and Mahalanobis teachers calibrate terminal and continuation
temperatures to their held-out empirical distance scale. Their listwise
ranking temperatures are calibrated to a shared target-entropy fraction.
Every cell reports target entropy and mean top-one margin.

## Gates and reporting

Training proceeds only after the preceding held-out gate passes: flat token
oracle, nested sentence rank/dynamics/symbolic purity, oracle sentence worker,
dynamic commutation, and grounded oracle high-level planning. Admissions bind
the exact checkpoint and held-out evaluation dataset; checkpoints separately
preserve their training-dataset lineage.

Final evaluation reports complete-solution accuracy on iGSM ID, near/far
length OOD, structural OOD, and paraphrase OOD. Planning depth varies
separately over `K0={8,16,32,64}` and `K1={1,2,4,8}`. Plots show accuracy
against candidate token/transition evaluations, and fixed `K0=32,K1=2` OOD
accuracy over value-training checkpoints. The depth evaluation is the full
Cartesian grid. It includes greedy frozen Qwen, an information-matched flat
token/value planner, and a prior-only manager. Candidate generation, exact
candidate grounding, token rollout, worker scoring, manager search, and exact
full-prefix re-encoding times are logged separately. The primary effort axis
is measured wall time per episode, with component token/transition counts and
Wilson accuracy intervals retained in the artifacts.

Controlled iGSM anchor/paraphrase/near-miss triplets provide quantitative
sentence-geometry evidence. Separate logical negation triplets, t-SNE, and
UMAP are qualitative out-of-domain views; they do not replace effective-rank,
triplet-accuracy, dynamics, or symbolic-purity gates.

The first submitted campaign uses one seed, 25% of the reference iGSM split
counts (50,000 training problems), 50,000 optimizer steps for every main
training stage, 64 examples per full-grid planning cell, and larger 128--256
example fixed-depth/gate evaluations. It is a
decision-grade single-seed mechanism run, not a multi-seed paper estimate.

The entry point is
[`scripts/run_full_hierarchical_language_experiment.py`](../../scripts/run_full_hierarchical_language_experiment.py).
Checkpoint-only search matrices use
[`scripts/run_hierarchical_mpc_search_matrix.py`](../../scripts/run_hierarchical_mpc_search_matrix.py).
