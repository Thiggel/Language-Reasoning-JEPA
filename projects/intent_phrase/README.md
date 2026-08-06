# Observed intent-phrase JEPA

The paper-facing method is now the terminal-safe, mixed-horizon Endpoint-Energy
JEPA specified in the [method handbook](method/README.md). The earlier
terminal-length-leaking scaling pilot is invalid; corrected training restores
the effect without that signal.

Current anonymous ICLR 2027 paper draft: [LaTeX source](paper/main.tex),
[compiled PDF](paper/main.pdf), [reproduction notes](paper/README.md), and
[internal reviewer audit](paper/REVIEW_AUDIT.md). The
draft uses the official 2027 style, includes all validated intent-phrase
results, labels one-seed and candidate-privileged evidence, and leaves missing
cross-domain and controlled-minimal-pair results visibly incomplete.

This is a first-class subproject for the controlled, small-scale reasoning
environment in which the available actions are natural-language intent
phrases. The model never generates the numerical outcome sentence: it selects
an intent, the environment executes it, and the JEPA predicts the resulting
latent reasoning state.

## Scientific scope

The project asks whether reconstruction-free latent dynamics can:

1. compute counterfactual consequences of observed language actions;
2. rank recursively imagined endpoints;
3. produce representations that retain the information needed for planning;
4. transfer from the controlled stylized generator to faithful iGSM.

Hierarchy is not part of the paper-facing claim. Corrected hierarchical
planning was negative and is retained as a documented limitation, not mixed
into the flat recipe.

## Current state

- The five-seed terminal-safe endpoint model reaches strict success
  `.120 +/- .012`, `.834 +/- .018`, `.874 +/- .029`, and `.877 +/- .029` at
  search depths `1,4,8,16`.
- The validated recipe uses recursive MLP prediction, mixed training horizons
  `{1,2,4,8}`, four sampled continuations per root, endpoint ranking, no dense
  rollout loss, and root-balanced beam search.
- Depth above one currently uses symbolic future feasible menus and is
  candidate-privileged.
- The primary open test removes the feasible menu and scores the complete
  action catalogue at every real and imagined state.
- Local GAR and hierarchy are controls, not primary method components.

## Navigation

- [Living method handbook](method/README.md): detailed, plain-text
  specifications of the task, model, GAR, multi-step teacher, planning,
  objectives, evaluation, controls, and representation analysis. Start here
  when checking what a term or experiment actually means.

- [Frozen ALFWorld pilot training-game identities](data/alfworld_pilot_train_games.jsonl),
  selected only for deterministic branch-complete collection before model
  evaluation; this fixture is not headline evaluation data.
- Intent-JEPA reasoning-state figure:
  [TikZ source](figures/intent_jepa_reasoning.tex),
  [rendered PDF](figures/intent_jepa_reasoning.pdf),
  [SVG](figures/intent_jepa_reasoning.svg)
- [Current status](STATUS.md)
- [GAR geometry and generative-bottleneck audit design](GAR_GEOMETRY_AUDIT.md)
- [Concise project, empirical, and representation-analysis report](../../research/reports/intent_phrase/2026-08-03-project-state-and-representation-analysis/REPORT.md)
- [Comprehensive project state, full result synthesis, and current method audit](../../research/reports/intent_phrase/2026-08-06-comprehensive-project-state/REPORT.md)
- [Sparse multidepth Endpoint-Energy factorial and launch audit](../../research/reports/intent_phrase/2026-08-06-multidepth-energy-factorial/REPORT.md)
- [Latest token prerequisite-support report](../../research/reports/intent_phrase/2026-07-21-token-prerequisite-support/REPORT.md)
- [ALFWorld implementation and admission status](../../research/reports/intent_phrase/2026-07-22-alfworld-admission/REPORT.md)
- [Discarded ALFWorld feasibility-hybrid diagnostic](../../research/reports/intent_phrase/2026-07-22-alfworld-overfit-failure/REPORT.md)
- [ALFWorld JEPA causal-supervision audit](../../research/reports/intent_phrase/2026-07-23-alfworld-jepa-supervision-audit/REPORT.md)
- [ALFWorld causal-prefix repair and remaining coverage gap](../../research/reports/intent_phrase/2026-07-23-counterfactual-prefix-repair/REPORT.md)
- [Full-catalogue versus oracle-feasible localization](../../research/reports/intent_phrase/2026-07-23-candidate-interface-localization/REPORT.md)
- [Code/config/run ownership](ARTIFACTS.md)
- [ICLR paper roadmap](PAPER_ROADMAP.md)
- [Geometry-first paper experiment contract](PAPER_EXPERIMENTS.md)
- [Detailed historical waves](../../research/intent_phrase/README.md)
- [Paper-facing experiment specification](../../research/intent_phrase/PAPER_PLAN.md)
- [Staged historical backlog](../../research/intent_phrase/BACKLOG.md)
- [Current causal matrix](../../research/intent_phrase/waves/12_causal_paper_matrix.md)
