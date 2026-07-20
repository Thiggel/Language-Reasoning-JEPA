# Cycle: global catalogue prior recovery

Status: implementation validated; bounded recovery planned

## Decision

Test whether the learned-catalogue planner failed because its behavioral prior
was trained only to rank currently feasible actions, leaving every infeasible
catalogue logit unconstrained at deployment.

## Observed result

Both learning-rate pilots completed without process errors. Across the 36
length-nine strict cells, 35 had zero success and invalid-action rate 1.0. The
only exception had success .033 and invalid-action rate .933. Nevertheless,
the top-four proposal set often retained useful roots: feasible precision was
.54--.75 and necessary-action recall was .69--.84. This localizes the failure
between candidate coverage and final root selection.

## Mechanism and correction

The availability head was trained over the whole prompt-derived action
catalogue, but the behavioral-prior cross entropy masked all infeasible
actions. At deployment, prior and availability scores rank the whole catalogue.
Thus infeasible prior logits could dominate despite never having competed in
the training denominator. The new `action_prior_candidate_scope=catalogue`
mode trains the demonstrated action against every real prompt action. The old
feasible-menu mode remains the default for historical protocols. This remains
explicit policy supervision, not action-free learning.

## Falsifiable recovery

Train only the availability and behavioral-prior heads at learning rates
3e-4, 1e-3, and 3e-3 from the same seed-zero checkpoint. Hold all encoder,
predictor, JEPA value, data, proposal-top-four, and beam-width-four choices
fixed. Evaluate lengths six and nine, strict and plus-two budgets, and depths
one, two, and four with matched prior-only and JEPA scoring.

Advance only if catalogue-wide top-one accuracy rises and selected invalid
actions fall materially (target below .25 at length nine). Only after that
gate may JEPA-versus-prior depth differences be interpreted. If invalid actions
remain dominant at all learning rates, stop training sweeps and redesign the
proposal factorization or class-balanced availability objective.

## Verification

The regression test first failed because the constructor and checkpoint gate
did not support catalogue scope. After implementation, focused tests pass,
the complete repository suite reports 120 passed, shell syntax passes, and a
checkpoint-initialized CPU smoke trained exactly the two intended heads with
finite losses before passing learned-catalogue checkpoint validation and
evaluation.

Human-facing result report:
[`../../reports/intent_phrase/2026-07-20-learned-catalogue-pilot-result/REPORT.md`](../../reports/intent_phrase/2026-07-20-learned-catalogue-pilot-result/REPORT.md).

