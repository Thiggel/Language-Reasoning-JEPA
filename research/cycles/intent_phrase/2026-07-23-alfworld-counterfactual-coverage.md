# ALFWorld counterfactual-coverage gate

## Decision

Do not admit paper-scale ALFWorld runs. Evaluate the best existing broad-
coverage checkpoint over depth/beam compute, while independently advancing
ProofWriter and PlanBench admission.

## Falsifiable question

Does increasing exact alternatives per factual state from two to twelve turn
stored transition learning into non-oracle full-catalogue closed-loop control?

## Result

No. Broader coverage improves counterfactual latent error, teacher-forced
median expert rank, and sometimes chosen-action feasibility, but every cell
still solves 0/8 training and 0/4 validation episodes. The lower learning rate
reduces validation invalid actions to 55.4%, still far above an admissible
controller.

## Next gate

Do not train another ALFWorld model yet. Test whether the existing checkpoint's
strong horizon-label fit produces any benefit at simulation depths 2--8 and
beam widths 1--8. In parallel, run source/schema/executor/reference/tiny-model
admission gates on the independent ProofWriter and three-block PlanBench
fixtures.

Report:
[`2026-07-23-alfworld-counterfactual-coverage`](../../reports/intent_phrase/2026-07-23-alfworld-counterfactual-coverage/REPORT.md)
