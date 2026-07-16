# Cycle: intent-phrase project boundary and ICLR roadmap

Status: complete; no new experiment round admitted

## Decision

Make observed intent-phrase JEPA a first-class repository subproject and
diagnose the causal preference-model gap before launching a broad recipe or
scale sweep.

## Evidence considered

- Common-protocol causal build-up and one-component ablations in Wave 12.
- Matched token/sentence policy baselines in Wave 00.
- Corrected negative hierarchy confirmations in Waves 10--11.
- The repaired counterfactual-outcome row currently in flight.

No unread human steering notes were present under
`.researchctl/steering/inbox/`, so no note changed this decision.

## Why no new plan was submitted

The active `research/NEXT_PLAN.json` already covers the repaired three-seed
counterfactual-outcome row. The next intent-phrase intervention depends on
whether existing diagnostics localize the gap to dynamics, teacher quality,
or student calibration. Submitting all three remedies simultaneously would
confound the result and duplicate paper-scale training before the current row
is terminal.

## Next falsifiable question

Does the causal J3 model trail the token policy because its transition model
is poorly optimized, or because the preference student fails to reproduce the
latent-goal teacher?

The smallest next step is an artifact-only stratified audit followed, only if
needed, by a causal context-window `{1,4,full}` and learning-rate cross-check.
The outcome patterns and scale gates are specified in
`projects/intent_phrase/PAPER_ROADMAP.md`.
