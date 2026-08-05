# Representation and geometry analysis

## Purpose

Planning accuracy shows whether a system works. Representation analysis asks
what information and structure make it work, what information is discarded,
and where the remaining failure originates.

The central theoretical prediction is local:

```text
planning success should track correct ordering of feasible successor actions
more strongly than it tracks global effective rank or one-step latent MSE.
```

## Local action-ordering analysis

For each state, compare every better action `a_plus` with every worse action
`a_minus`.

Report:

- fraction satisfying `E(a_plus) < E(a_minus)`;
- top-1 optimal-action recall;
- regret of the selected action;
- rank correlation between Energy and exact continuation quality;
- margin `min_negative_energy - best_positive_energy` under a consistent sign
  convention;
- how these statistics change with imagined rollout depth.

Across checkpoints and seeds, correlate these quantities with strict success.
Repeat the same analysis for latent prediction error, effective rank, and probe
scores. This can falsify the claim that GAR's local ordering is the important
representation property.

## Progress and residual structure

A working hypothesis is:

```text
z = z_progress + z_residual
```

`z_progress` may be a small goal-aligned component controlling action order,
while `z_residual` retains operation, entity, and predicted-consequence detail.

Test prediction of remaining distance from:

- scalar latent distance to the goal;
- GAR Energy;
- full latent state;
- latent residual after removing the learned progress direction.

Control for elapsed step index. Resolved count and step number are correlated in
ordinary trajectories, so a progress probe can otherwise learn position rather
than semantic progress.

## Controlled language minimal pairs

Primary tests should manipulate one semantic or surface factor at a time:

- same semantic state, different paraphrase;
- same action semantics, different action phrase;
- high lexical overlap with changed negation;
- operator swap such as plus versus minus;
- same operation word in a different causal role;
- irrelevant entity renaming;
- matched step index with different true remaining progress.

Measure full-space distance, nearest-neighbor retrieval, successor similarity,
Energy change, and action-ranking preservation.

## Linear probes

Freeze representations and fit simple linear predictors for:

- operation identity;
- resolved variable count;
- remaining necessary steps;
- current step index;
- whether an action is necessary;
- exact numerical outcome;
- entity or dependency properties where labels are available.

Use balanced accuracy for imbalanced binary labels. A probe at chance is not
positive evidence, even if ordinary accuracy appears high.

## Nonlinear probes

A small nonlinear probe distinguishes “information absent” from “information
present but not linearly organized.” It should remain low-capacity and use the
same train/validation split as the linear probe.

## Reconstruction

Frozen-feature sentence reconstruction asks whether surface wording survives.
It is secondary because inability to reproduce arbitrary wording can be
desirable under the predictive-quotient story.

More interpretable reconstruction targets are:

- semantic action identity;
- canonicalized operation and arguments;
- numerical result;
- controlled paraphrase class.

## Dimensionality and collapse diagnostics

Report:

- per-feature standard deviation;
- covariance spectrum;
- effective rank;
- duplicate-state similarity;
- variance across problems and steps.

These diagnose collapse. They do not explain planning quality. A low-rank token
model can still plan well, and a high-rank JEPA can still order actions poorly.

## PCA, t-SNE, and UMAP

Use these only as illustrations after quantitative tests.

Plots should:

- use fixed, regeneratable scripts and stored coordinates;
- compare the same examples across models;
- label paraphrase, negation, operator, and progress groups;
- avoid interpreting global distance in t-SNE as metric evidence;
- omit plot titles when the paper caption supplies the explanation;
- show uncertainty or multiple seeds where feasible.

## GAR mechanism analysis

Compare:

- full GAR;
- GAR with geometry gradients detached;
- geometry-only planning without the head;
- direct ranker;
- counterfactual-only data control.

If full GAR improves local ordering and transfer beyond the direct ranker, that
supports the predictive-geometry mechanism. If only the head improves, the
result is better described as supervised scoring over predictive features.
