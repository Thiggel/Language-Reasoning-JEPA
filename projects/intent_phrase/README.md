# Observed intent-phrase JEPA

This is a first-class subproject for the controlled, small-scale reasoning
environment in which the available actions are natural-language intent
phrases. The model never generates the numerical outcome sentence: it selects
an intent, the environment executes it, and the JEPA predicts the resulting
latent reasoning state.

## Scientific scope

The project asks whether reconstruction-free latent dynamics can:

1. compute counterfactual consequences of observed language actions;
2. learn which action advances a reasoning problem without symbolic ranking;
3. produce representations that retain the information needed for planning;
4. transfer from the controlled stylized generator to faithful iGSM.

Hierarchy is not part of the paper-facing claim. Corrected hierarchical
planning was negative and is retained as a documented limitation, not mixed
into the flat recipe.

## Current state

- The matched token intent policy remains the strongest non-oracle baseline:
  `.827 +/- .003` strict and `.978 +/- .003` with two extra actions.
- The older reduced non-symbolic JEPA reached `.797 +/- .008/.963 +/- .008`,
  but used the previous predictor/protocol and is not the frozen paper model.
- The clean causal-transformer build-up reaches `.588 +/- .013/.845 +/- .043`
  after two-step latent-goal preference distillation.
- Faithful action-displacement decoding and terminal-distance monotonicity are
  promising individual add-backs (`.632` and `.637` strict respectively), but
  no combined recipe has been validated.
- Dense four-step rollout and residual prediction are negative in the current
  causal matrix.
- The causal counterfactual-outcome ablation is being rerun after correcting a
  batch/time indexing bug in independent alternative-action prefixes.

The immediate scientific problem is therefore not “add hierarchy.” It is to
explain and close the causal model's action-selection gap under a frozen,
information-matched protocol.

## Navigation

- [Intent-JEPA reasoning-state figure](figures/intent_jepa_reasoning.svg)
- [Current status](STATUS.md)
- [Latest token prerequisite-support report](../../research/reports/intent_phrase/2026-07-21-token-prerequisite-support/REPORT.md)
- [Code/config/run ownership](ARTIFACTS.md)
- [ICLR paper roadmap](PAPER_ROADMAP.md)
- [Geometry-first paper experiment contract](PAPER_EXPERIMENTS.md)
- [Detailed historical waves](../../research/intent_phrase/README.md)
- [Paper-facing experiment specification](../../research/intent_phrase/PAPER_PLAN.md)
- [Staged historical backlog](../../research/intent_phrase/BACKLOG.md)
- [Current causal matrix](../../research/intent_phrase/waves/12_causal_paper_matrix.md)
