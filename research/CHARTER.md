# TextJEPA research charter

## Objective

Determine whether a joint-embedding predictive architecture can learn useful
hierarchical actions for language: token transitions at the base level and
predictive state transitions over spans, phrases, sentences, and eventually
paragraphs at higher levels. The result must improve prediction, planning, or
generation for reasons attributable to the hierarchy—not to target leakage,
candidate information, ordering artifacts, or privileged symbolic labels.

The active repository evidence is narrower than this long-term objective. It
contains controlled observed-intent iGSM studies and a hard-text track without
intent annotations. Preserve that separation. Do not describe an observed
intent phrase as a discovered language action, and do not treat latent planning
success as free-form language generation.

## Scientific standard

Optimize reliable information gained per unit of elapsed time. Every cycle
must name one concrete decision, a falsifiable question, the result patterns
that would change that decision, the smallest faithful test, validity gates,
and the evidence required to scale. Idle GPUs are not evidence that more jobs
are useful.

For every mechanism, challenge:

- target leakage and candidate-information mismatch;
- representation collapse, effective rank, variance, covariance, and target
  predictability;
- optimization and capacity limits in the encoder, predictor, action/span
  representation, and evaluator;
- masking, boundaries, packing, trajectory length, within-batch diversity,
  and data mixture;
- target-encoder dynamics and stop-gradient placement;
- whether a non-hierarchical information-matched control explains the result;
- whether the metric measures semantic/generative utility or only probe
  decodability;
- ordering, truncation, seed, and evaluator artifacts already observed in this
  repository.

Tune alternatives fairly. VICReg, SIGReg, and other stabilizers do not share a
meaningful universal coefficient range. Use a coarse stability-finding sweep
for each method, then a small refinement around viable regions. Sweep learning
rate when optimization is a plausible interaction, not as ritual.

## Scale gates

Increase model size, sequence length, data volume, or seed count only after:

1. the implementation passes synthetic/unit tests for masks, targets, losses,
   metrics, and causality;
2. the measurement distinguishes healthy prediction from collapse and
   leakage;
3. matched controls and at least one negative control behave as expected;
4. the pilot result is large enough and stable enough to affect a decision;
5. the larger run has a predeclared paper-relevant purpose.

Paper-scale training additionally requires a frozen protocol, fixed primary
metrics, predeclared exclusions, adequate seeds, a compute budget approved by
a human, and reproduction on a second backend.

## Stop or redirect conditions

Materially change or abandon a hierarchy when corrected, information-matched
evaluations repeatedly show no benefit; the apparent benefit depends on
privileged labels or proposal ordering; representations remain collapsed
across reasonable per-method tuning; or useful span targets require boundaries
not recoverable at inference. Record negative results rather than silently
recycling the same hypothesis under a new name.

## Human-only decisions

Humans approve charter changes, budget increases, multi-node or paper-scale
campaigns, new datasets or large transfers, external publication, credential
changes, destructive cleanup, and cancellation of jobs owned outside the
controller.

