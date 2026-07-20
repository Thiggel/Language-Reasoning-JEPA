# Cycle: learned action catalogue for deployable simulation

Status: implementation validated; bounded pilot pending submission

## Decision

Determine whether a fixed prompt-derived action catalogue, learned future availability, and a behavioral prior can support depth-two/four JEPA planning without current or future symbolic feasible-action menus.

## Falsifiable question

Holding the catalogue, proposal heads, top-four roots, root-balanced beam width four, maximum expansion, problems, and evaluation budgets fixed, does JEPA trajectory scoring improve strict success over accumulated prior-only proposal scoring as simulation depth increases from one to two or four?

## Result patterns that change direction

- Advance to three seeds if learned proposals have healthy feasible/necessary recall and JEPA improves matched strict success by at least .05 or exhibits a clear positive depth trend.
- If proposal recall or invalid selection is poor, stop the JEPA comparison and repair availability/proposal calibration.
- If proposals are healthy but JEPA remains worse at every depth, retain the prior and diagnose value calibration versus rollout drift.
- Do not change GAR beam width in this round. Once proposal validity passes, compare GAR teacher beam widths 1, 4, and 8 with the selected planner protocol.

## Minimal experiment

Adapt only the availability and prior heads from the existing seed-zero action-prior checkpoint. Compare head learning rates 3e-4 and 1e-3. Evaluate availability weights 0.5, 1, and 2; depths 1, 2, and 4; and prior-only versus JEPA scoring on identical length-six/nine problems at strict and plus-two budgets.

## Validity gates

- stored checkpoint configuration passes the learned-catalogue eligibility gate;
- no oracle-future-action flag;
- no symbolic simulator or oracle goal;
- top-four roots and per-root beam width four, bounded by global expansion;
- invalid actions reported without fallback;
- finite head losses and metrics;
- state encoder, action encoder, predictor, value, and target encoders remain frozen;
- full repository tests pass before snapshot submission.

## Implementation evidence

Three test-first loops produced expected failures before each implementation layer. Targeted planner/model tests pass, a real-checkpoint evaluation smoke completes, a tiny checkpoint-initialized head-training smoke trains 0.28M intended parameters with finite losses, and the final repository suite reports 119 passed tests.

Human-facing implementation report:
[`../../reports/intent_phrase/2026-07-20-learned-action-catalogue-implementation/REPORT.md`](../../reports/intent_phrase/2026-07-20-learned-action-catalogue-implementation/REPORT.md).

