# Cycle: causal J3 learning-rate screen

Status: seed-0 screen complete; two-seed confirmation selected

## Decision

Confirm only the full-history J3 learning rate `1e-3` with seeds 1 and 2.
Seed 0 reached `.705` strict success versus `.590` for the matched `3e-4`
reference, clearing the predeclared `+.05` gate. Context 1 (`.155`), context 4
(`.475`), and `1e-4` (`.295`) did not pass.

## Falsifiable question and rule

Does the `1e-3` optimization change produce a stable multi-seed improvement,
rather than a favorable seed-0 fluctuation? Promote it only if the three-seed
mean strict success exceeds the existing J3 mean `.588` by at least `.05`, at
least two of three seeds improve over their matched reference seeds, teacher
top-1 remains at least `.80`, and state standard deviation/effective rank show
no collapse. Otherwise retain `3e-4` and redirect to preference/deployment
calibration. No final test is opened.

## Audited evidence and validity

All four completed cells used shuffled feasible intent menus, 200 validation
problems, 100 privileged symbolic-oracle audit anchors, 13,509 probe
transitions, and healthy state rank (`242.3`--`246.9`) and variance
(`1.035`--`1.039`). The `1e-3` cell also reached `.920` slack-two success and
`.409` one-step value decodability, but recursive value decodability was only
`.246`; the result is behavioral, not proof that rollout fidelity was fixed.
The original four launches, two three-hour timeouts, and the human-approved
five-hour cancellation are process-invalid and excluded without being hidden.

## Steering effects

The ICLR-roadmap note forces the `.827` matched token-policy result to remain
the paper gate; even the seed-0 winner is not a JEPA victory. The overnight
note required infrastructure-safe, small, decision-relevant recovery and no
duplicate Alex job; short temporary paths enabled valid runs, all failures
remain visible, and the next plan contains only two confirmation jobs. The
supplied allocation permits this bounded confirmation within 16 GPU-hours.

Report: `research/reports/intent_phrase/2026-07-17-j3-learning-rate-screen/REPORT.md`.
