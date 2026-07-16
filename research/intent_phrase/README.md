# Observed-intent-phrase JEPA

## Paper role

Use faithful iGSM for headline results. Retain stylized iGSM as a fast,
fully controlled development domain and as an appendix/diagnostic suite. The
small domain is valuable for exhaustive counterfactual audits, causal probes,
and architecture screening, but it should not be presented as the final
language benchmark.

Frame this project as exploratory evidence that action-conditioned latent
world models can represent, predict, and plan over language reasoning. The
paper should show a cumulative scientific build-up after the hierarchy,
regularizer, rollout, and GAR screens identify their best settings. Historical
fixed-point shearing remains diagnostic evidence; no additional shearing wave
is planned.

## Current reference before hierarchy selection

The provisional non-symbolic reference uses direct one-step dynamics, an EMA
target, VICReg, on-trajectory outcome prediction and predicted-outcome
consistency, and H=2/K=2 geometric preference distillation. Three-seed
stylized validation is 0.797 +/- 0.008 strict and 0.963 +/- 0.008 with two
extra actions. It is not frozen and is not yet the paper's final model.

## Experiment waves

1. [`00_baselines_and_protocol.md`](waves/00_baselines_and_protocol.md)
2. [`01_representation_and_dynamics.md`](waves/01_representation_and_dynamics.md)
3. [`02_geometry_and_symbolic_selection.md`](waves/02_geometry_and_symbolic_selection.md)
4. [`03_nonsymbolic_selection.md`](waves/03_nonsymbolic_selection.md)
5. [`04_horizon_search_and_hierarchy.md`](waves/04_horizon_search_and_hierarchy.md)
6. [`05_faithful_igsm_transfer.md`](waves/05_faithful_igsm_transfer.md)
7. [`06_observed_action_probabilistic.md`](waves/06_observed_action_probabilistic.md)
8. [`07_representation_probes.md`](waves/07_representation_probes.md)
9. [`08_hierarchical_planning.md`](waves/08_hierarchical_planning.md)
10. [`09_hierarchy_support_and_value.md`](waves/09_hierarchy_support_and_value.md)
11. [`10_dense_rollout_confirmation.md`](waves/10_dense_rollout_confirmation.md)
12. [`11_controller_outcomes_and_discrete_hierarchy.md`](waves/11_controller_outcomes_and_discrete_hierarchy.md)

The staged future plan is in [`BACKLOG.md`](BACKLOG.md). Raw logs are indexed
in [`logs/README.md`](logs/README.md). The frozen easy-domain section and
three-seed completion matrix are in [`PAPER_PLAN.md`](PAPER_PLAN.md).
