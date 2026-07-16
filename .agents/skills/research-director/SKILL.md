---
name: research-director
description: Reconstruct the compact TextJEPA evidence state and choose the next highest-value scientific decision. Use when starting or continuing a TextJEPA research cycle, deciding what to test next, auditing whether more experiments are justified, or preparing research/NEXT_PLAN.json.
---

# Direct one TextJEPA research cycle

Read `research/CHARTER.md`, `STATE.md`, `EVIDENCE.md`,
`QUESTION_BACKLOG.md`, `EXPERIMENT_INDEX.md`, the current cycle linked from
STATE, the latest cluster inventory, and newly completed run summaries. Do not
recursively load historical waves or raw logs; follow a source link only when
it bears on the current decision.

Audit scientific validity before interpreting metrics. Separate observation,
inference, and speculation. Generate several concrete decision questions,
then rank them by expected decision relevance, uncertainty reduction,
interpretability, elapsed time, and GPU-hours. Select one decision, not a bag
of loosely related experiments.

Use `$literature-review` only for a narrow uncertainty that could change the
design. Use `$experiment-designer` to turn the selected question into a
falsifiable minimal experiment. Use `$results-analyst` when terminal results
exist. Use `$explain-research` before considering the cycle complete or writing
the next plan. Use `$beamer-synthesis` for a slide companion after the report
passes validation.

Read unhandled notes under `.researchctl/steering/inbox/<project>/` before
choosing the next decision and state explicitly how each note affected it.
Update one cycle document and the compact ledgers. Implement only code needed
for the selected decision and test causality, targets, masks, losses, and
metrics where applicable. Finish with `research/NEXT_PLAN.json` conforming to
`automation/schema/run-plan.schema.json`, or record why no experiment is
justified. Do not submit jobs directly.
