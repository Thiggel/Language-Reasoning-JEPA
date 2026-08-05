# Evaluation and success metrics

## Closed-loop success

An episode starts with a generated problem and ends when either:

- the queried variable is resolved;
- the action budget is exhausted.

The main outcome is whether the problem was solved.

## Necessary length and excess actions

Let `L_star` be the minimum number of necessary actions for a problem. If the
model solves in `L_model` actions, define excess actions:

```text
excess = L_model - L_star
```

Strict success means `excess=0`.

Slack-k success means:

```text
excess <= k
```

The current runners evaluate slack 0 and slack 2 separately. Because the policy
does not observe its remaining evaluation budget, this is computationally
redundant. A better evaluator should run once with a larger maximum budget,
record the first solution time, and derive the whole cumulative curve:

```text
accuracy_at_slack(k) = fraction of episodes with excess <= k
```

It should also report the exact excess histogram for 0, 1, 2, 3, and so on.

## Current planning matrix

The Energy screen evaluates every checkpoint with:

- 200 validation problems;
- beam width 8;
- planner depths 1, 4, 8, and 16;
- slack 0 and 2;
- identical validation seed and generated puzzle order.

This yields paired puzzle sets but only aggregate summaries are currently
stored. Paper runs should preserve per-episode outcomes so paired confidence
intervals and disagreement analysis can be computed directly.

## Interpretation of depth

Depth 1 uses the current feasible menu only.

Depth above one uses symbolic future feasible menus and must be labeled
candidate-privileged. It measures how well learned consequence simulation and
Energy-based valuation exploit an externally supplied action tree.

## Metrics currently reported

- **Success:** fraction solved within the budget.
- **Mean steps:** average number of executed actions, including truncated
  failures.
- **Mean necessary:** average shortest necessary length of the evaluated
  problems.
- **Distractor rate:** fraction of executed actions not required by the exact
  solution dependency set.

## Metrics that should be added

### Action-ordering metrics

- pairwise feasible-action ranking accuracy;
- top-1 optimal-action recall;
- mean action regret;
- Spearman or Kendall correlation with exact continuation quality;
- positive-versus-best-negative Energy margin.

### Planning-geometry metrics

- correlation of ranking accuracy with strict success across checkpoints;
- correlation of margin with strict success;
- corresponding correlations for latent MSE, effective rank, and probe scores;
- calibration error of predicted Energy versus teacher and exact quality;
- degradation as imagined-state rollout depth grows.

### Statistical reporting

- five training seeds for final results;
- at least 1,000 paired evaluation problems for close comparisons;
- paired bootstrap confidence intervals for model differences;
- explicit hyperparameter-search budget per model family.

## Length out-of-distribution evaluation

The final paper should train on a bounded reasoning length and evaluate on
longer lengths without retraining. Report success as a function of necessary
length, not only one pooled OOD number.

For example:

```text
x-axis: necessary reasoning length
y-axis: strict or slack-k accuracy
curves: JEPA, token LM, sentence LM, hybrid
```

This distinguishes memorizing the training-length regime from learning a
transition representation that composes.

## Validation versus test split

Recipe selection, loss-weight choice, teacher horizon, beam composition, and
learning rate must use validation problems. The held-out test split should be
opened only after those choices are frozen.

## Checkpoint selection

Current jobs save `best.pt` by the model's own composite validation objective.
Before paper-scale repetition, compare `best.pt` and `last.pt` planning results
to ensure the conclusion is not caused by method-specific checkpoint selection.

## Oracle and symbolic labels in tables

Every table or figure should visibly tag:

- symbolic current feasible menu;
- candidate-privileged future action tree;
- oracle encoded terminal goal;
- exact environment transition;
- exact remaining-step distance.

These are useful controlled diagnostics but answer narrower questions than an
unrestricted language agent.
