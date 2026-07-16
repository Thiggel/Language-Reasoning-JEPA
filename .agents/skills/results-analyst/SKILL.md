---
name: results-analyst
description: Audit completed TextJEPA runs and produce defensible tables, figures, conclusions, and next-step evidence. Use when a research round completes, metrics disagree, runs fail or are incomplete, exclusions must be decided, or cycle/evidence documents need updating.
---

# Analyze a completed round

Read the resolved plan, manifests, compact run summaries, resolved configs,
environment records, and declared artifacts. Open raw logs only for failed,
invalid, suspicious, or metric-inconsistent runs.

First classify process state and scientific validity separately. Check planned
commit/config/data/seed identity, completion, NaNs, collapse diagnostics,
ordering and candidate-information controls, sample counts, exclusions, and
missing artifacts. Never silently drop a run; record the reason.

Produce a compact paired table with uncertainty appropriate to the design and
figures that expose distributions or failure strata rather than only headline
means. Use intuitive labels. Compare only information- and protocol-matched
systems; label oracle or privileged-candidate diagnostics explicitly.

Write observations, inferences, speculation, limitations, and the predeclared
decision outcome separately. Update the cycle, `EVIDENCE.md`,
`EXPERIMENT_INDEX.md`, and `DECISIONS.md` as warranted. Do not propose another
round until the current result's validity is settled. Invoke
`$explain-research` to create the human-facing report; a terse cycle ledger is
not a substitute for that report.
