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
Its first predicted successor is the sentence waypoint. The frozen Qwen LM
proposes complete token spans, `P0` predicts their endpoints, and the worker
selects text by coarse waypoint cost plus LM prior cost. After execution, the
frozen LM is run again and all imagined states are discarded. This is exact
hierarchical receding-horizon control.

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
accuracy over value-training checkpoints. Candidate generation, manager,
worker, and exact re-encoding wall times are logged separately.

The entry point is
[`scripts/run_full_hierarchical_language_experiment.py`](../../scripts/run_full_hierarchical_language_experiment.py).
