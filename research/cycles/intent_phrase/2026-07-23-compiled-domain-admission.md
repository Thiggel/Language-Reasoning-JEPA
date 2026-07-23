# Compiled-domain admission and full-catalogue localization

## Decision

Do not admit ProofWriter or PlanBench learning-rate sweeps yet. Test whether
exhaustive observed consequences for the same full catalogue repair
teacher-forced rank, invalid-action selection, and tiny-set closed-loop fit.

## Falsifiable question

Did the first compiled-domain gate fail because the JEPA could not learn stored
transitions, or because deployment scored unsupervised infeasible actions?

## Result

The second explanation is supported. Stored configured-horizon GAR top-1 is
96.3% on ProofWriter and 94.4% on PlanBench, while full-catalogue expert top-1
falls to 33.3% and 11.8%. Invalid actions had no compiled consequences. The
nominal shuffle control was invalid because the observed-action loader ignored
its flag.

## Next gate

Repeat the same 24/8/8 tiny gate with every public-catalogue action assigned its
executor-observed consequence, including unchanged-state invalid outcomes.
Run aligned learning rates 0.001 and 0.003 plus the repaired shuffle at 0.001.

Report:
[`2026-07-23-compiled-domain-admission`](../../reports/intent_phrase/2026-07-23-compiled-domain-admission/REPORT.md)
