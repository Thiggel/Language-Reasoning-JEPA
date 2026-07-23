# Candidate-interface localization

## Decision

Retain the full grounded catalogue as the paper headline, and evaluate every
final checkpoint against the current oracle-feasible subset as an explicitly
candidate-privileged diagnostic.

## Falsifiable question

Are the compiled-domain zero-solve results caused mainly by invalid-action
selection, or does the model also fail to order actions when feasibility is
provided?

## Result

Feasibility is the main bottleneck but not the only one. At +4 actions,
ProofWriter rises from 0% to 87.5% and PlanBench from 12.5% to 50% under the
feasible-only interface. ProofWriter strict success remains at most 12.5%.
The PlanBench shuffled-action control remains 0% feasible-only, supporting a
real action-grounding effect.

## Next action

Start the independent official-iGSM model/width learning-rate campaign under
the frozen dual-interface evaluation. Scale the compiled domains only after
moving beyond the 24/8 tiny fixture.

Report:
[`2026-07-23-candidate-interface-localization`](../../reports/intent_phrase/2026-07-23-candidate-interface-localization/REPORT.md)
