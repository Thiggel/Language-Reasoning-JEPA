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

Earlier shared-state hierarchy is not a positive paper-facing claim: corrected
planning was negative.  A new, explicitly exploratory distinct-state hierarchy
is now being tested after token-level work exposed two necessary validity
conditions: complete low paths must be lifted into separate high coordinates,
and causal high planning must retain its K-stride history.  Its direct
remaining-step value labels are symbolic training supervision and are reported
separately from label-free results.

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
- Exact dense recursive rollout, deterministic EMA targets, parameter scale,
  GAR breadth plus advantage regression, and distinct-state hierarchy are in
  an active J3 recipe screen; none is yet an established improvement.
- The causal counterfactual-outcome ablation is being rerun after correcting a
  batch/time indexing bug in independent alternative-action prefixes.

The immediate scientific problem is to identify a healthy J3 optimization and
capacity regime, then test whether dense recursion, GAR calibration, or a
coordinate-correct hierarchy closes the action-selection gap under frozen,
information-matched planning protocols.

## Navigation

- [Current status](STATUS.md)
- [Active weekend recipe cycle](../../research/cycles/intent_phrase/2026-07-18-weekend-recipe-search.md)
- [Code/config/run ownership](ARTIFACTS.md)
- [ICLR paper roadmap](PAPER_ROADMAP.md)
- [Latest terminal-run audit report](../../research/reports/intent_phrase/2026-07-16-terminal-run-validity-audit/REPORT.md)
- [Action-prior and hierarchy planning report](../../research/reports/intent_phrase/2026-07-20-action-prior-hierarchy-planning/REPORT.md)
- [Learned action-catalogue implementation audit](../../research/reports/intent_phrase/2026-07-20-learned-action-catalogue-implementation/REPORT.md)
- [Learned action-catalogue pilot result](../../research/reports/intent_phrase/2026-07-20-learned-catalogue-pilot-result/REPORT.md)
- [Explicit action-history support implementation](../../research/reports/intent_phrase/2026-07-20-explicit-action-history-support/REPORT.md)
- [Detailed historical waves](../../research/intent_phrase/README.md)
- [Paper-facing experiment specification](../../research/intent_phrase/PAPER_PLAN.md)
- [Staged historical backlog](../../research/intent_phrase/BACKLOG.md)
- [Current causal matrix](../../research/intent_phrase/waves/12_causal_paper_matrix.md)
- [Current calibrated proposal/JEPA reranking cycle](../../research/cycles/intent_phrase/2026-07-20-hybrid-proposal-jepa-reranking.md)
- [Current token-level prerequisite-support cycle](../../research/cycles/intent_phrase/2026-07-21-token-prerequisite-support.md)
- [Hybrid result and token-support report](../../research/reports/intent_phrase/2026-07-21-hybrid-reranking-and-token-support/REPORT.md)
