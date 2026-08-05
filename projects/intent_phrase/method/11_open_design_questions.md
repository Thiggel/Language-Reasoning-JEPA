# Open design questions

This file records choices that are not yet settled. It prevents an experimental
runner from silently becoming the method definition.

## 1. What exactly should transition Energy predict?

Current default:

```text
H-step geometric advantage target
  = endpoint distance after root-balanced teacher continuation
    - current distance
```

Alternatives:

- absolute endpoint distance;
- exact environment continuation cost as an oracle-supervised control;
- one-step potential difference only;
- Bellman-consistent learned cost-to-go.

Decision needed: whether the paper calls the target “advantage,” “progress,” or
“Energy” in each context.

## 2. How should an H-step transition Energy be used at test time?

Current intended rule: terminal-only Energy of the final beam transition.

Controls:

- cumulative edge Energies, now considered a negative control;
- root-only H-step Energy, equivalent to distilled one-step selection;
- terminal state Energy.

The terminal-composition evaluation launched on 2026-08-05 decides whether the
previous collapse was primarily caused by cumulative composition.

## 3. Should H=1 use imagined or true endpoints?

Current implementation:

- H=1 target uses true one-step EMA-encoded outcomes;
- H>1 target uses EMA-predictor endpoints.

This makes the horizon sweep slightly more than a horizon-only intervention.
A strict control should use predicted H1 endpoints or true-transition search at
all tractable horizons.

## 4. Is the geometry teacher sufficiently close to optimal?

Current teacher: root-balanced beam width four, scored by EMA goal distance.

Needed diagnostics:

- compare with exhaustive geometry search on short problems;
- sweep teacher width;
- compare geometry-selected and exact-quality-selected continuations;
- measure root label stability across EMA checkpoints.

## 5. MLP or causal Transformer as the main JEPA?

The current horizon result uses the historical MLP predictor. The intended
architecture is causal. The result must be reproduced with a capacity- and
training-matched causal predictor before making an architecture-general claim.

## 6. How should action availability be represented?

Current headline laboratory protocol supplies the symbolic current feasible
menu. D>1 additionally supplies symbolic future menus.

Longer-term options:

- learn feasibility separately;
- propose from the full action catalogue;
- retrieve top-M language actions and rerank with JEPA;
- generate unrestricted actions.

These broaden the task and should not obscure the controlled geometry claim.

## 7. How should slack be evaluated?

Current runners repeat slack 0 and 2 evaluations.

Preferred design: one rollout with a maximum excess-action budget, recording
first solution time and deriving the full exact and cumulative excess curves.

## 8. Which result deserves five seeds?

Scale gate:

- training and evaluation complete without collapse;
- candidate and outcome privileges are labeled;
- corrected terminal planning is implemented;
- the effect is larger than 200-episode sampling uncertainty;
- no nearby learning-rate control reverses the conclusion.

Likely candidates are the best H8/H16 transition Energy, the H1 reference, the
state-Energy control, the direct ranker, and the strongest sentence baseline.

## 9. What second domain should be used?

Candidates:

- textual BlocksWorld;
- relational graph navigation;
- logic-dependency planning;
- ALFWorld after its supervision and action-interface validity gates pass.

The second domain should preserve discrete language actions and exact local
counterfactual evaluation without duplicating arithmetic dependency structure.

## 10. Which representation analyses are primary?

Recommended order:

1. local action ranking and margins;
2. rollout-depth degradation;
3. paraphrase, negation, and operator minimal pairs;
4. progress-versus-step deconfounding;
5. probes and effective rank;
6. t-SNE or UMAP illustrations;
7. surface reconstruction.

## Proposed collaboration rule

When we change an answer, record:

- the old definition;
- the new definition;
- why it changed;
- code/configs affected;
- which old results remain valid;
- which results need reevaluation.
