---
name: research-director
description: Reconstruct the compact TextJEPA evidence state and choose the next highest-value scientific decision. Use when starting or continuing a TextJEPA research cycle, deciding what to test next, auditing whether more experiments are justified, or preparing research/NEXT_PLAN.json.
---

# Direct one TextJEPA research cycle

Read only the compact project state and newly completed summaries needed for
the current decision. Reports, historical cycles, global ledgers, and steering
notes are optional context and must not block experiments.

Audit scientific validity before interpreting metrics. Separate observation,
inference, and speculation. Generate several concrete decision questions,
then rank them by expected decision relevance, uncertainty reduction,
interpretability, elapsed time, and GPU-hours. Select one decision, not a bag
of loosely related experiments.

Use `$literature-review` only for a narrow uncertainty that could change the
design. Use `$experiment-designer` to turn the selected question into a
falsifiable minimal experiment. Use `$results-analyst` when terminal results
exist. Use `$explain-research` or `$beamer-synthesis` only when the human asks
for explanatory artifacts or they materially help a decision; neither is a
prerequisite for the next plan.

Consider relevant notes under `.researchctl/steering/inbox/<project>/` when
useful. Cycle and ledger updates are optional. Implement only code needed
for the selected decision and test causality, targets, masks, losses, and
metrics where applicable. Finish with `research/NEXT_PLAN.json` conforming to
`automation/schema/run-plan.schema.json`, or record why no experiment is
justified. Use the controller for cluster submission so round identity and job
tracking remain coherent.
