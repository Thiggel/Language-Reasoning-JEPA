# Endpoint-Energy JEPA paper specification

This directory defines the paper-facing intent-phrase method. The normative
recipe is the simplest validated system: recursive latent prediction,
horizon-conditioned endpoint Energy, and receding-horizon beam search. Local
GAR is not part of the primary method.

Implementation reference: commit `5615e47`. Headline evidence uses the
terminal-safe checkpoints trained at commit `87ebfad`.

## Reading order

1. [Task and interfaces](01_task_and_interfaces.md)
2. [Model](02_model_and_latent_states.md)
3. [Endpoint Energy](03_gar.md)
4. [Training targets](04_multistep_energy_teacher.md)
5. [Planning](05_planning.md)
6. [Objectives](06_training_objectives.md)
7. [Evaluation](07_evaluation.md)
8. [Baselines and ablations](08_baselines_and_controls.md)
9. [Representation analysis](09_representation_analysis.md)
10. [Notation](10_notation_and_glossary.md)
11. [Open decisions](11_open_design_questions.md)

## Paper claim

> An action-conditioned JEPA can reason in controlled language environments by
> recursively imagining language-action consequences and ranking their
> endpoints. More test-time search improves planning when endpoint Energy is
> trained at multiple horizons.

This is a controlled study of predictive representations. It is not an
unrestricted language-reasoning agent.

## Validated default

| Choice | Value |
|---|---|
| State width | 256 |
| Action width | 16 |
| Predictor | residual concatenation MLP |
| Training horizons | sampled from `{1,2,4,8}` |
| Horizon input to Energy head | none (horizon-blind) |
| Rollouts per root | 4 |
| Alternative roots | 2 plus the factual root |
| Endpoint loss | logistic pairwise ranking |
| Dense rollout loss | 0 |
| Dropout | 0 |
| Examples | 300,000 fresh generated problems |
| Test search | root-balanced beam, width 8 |
| Test depths | `1,2,4,8,16` |
| Beam score | final endpoint Energy only |

Frozen-recipe (horizon-blind Energy, 2026-08-07) five-seed strict success is
`.126, .450, .837, .879, .884` at depths `1,2,4,8,16`
(slack-2 `.541, .863, .979, .991, .995`). The earlier horizon-conditioned
mixed-horizon result (`.120, .834, .874, .877` at depths `1,4,8,16`) is
superseded: the Energy head no longer receives any horizon input, which
removes every out-of-support depth query, and the 0.25 root pair-difference
auxiliary is retained (coherent: one shared Energy semantics).

## Independent depths

- `H`: sampled training rollout horizon.
- `R`: number of sampled training rollouts per root.
- `D`: test-time beam depth.
- `B`: beam width retained per first-action root.

A training horizon and a test depth are not the same variable.
