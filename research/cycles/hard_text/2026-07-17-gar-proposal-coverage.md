# Cycle: GAR proposal coverage after counterfactual saturation

## Decision

Stop increasing the primitive counterfactual count. Test whether the geometric
advantage ranker is limited by prior-only candidate coverage rather than by the
number of candidates: at fixed K=32, compare prior-only, half-prior/half-random,
and random full-vocabulary proposals.

## Evidence audited

All six counterfactual-budget processes completed, but their controller
summaries contained no metrics and still said scientific validity was not
assessed. The explicitly expected compact `metrics.json` and planner-interface
audits were therefore used; no raw log was opened. Across K=4, 8, 32, 64, and
94, terminal held-out support-pair accuracy rose from 0.818 to 0.902, while
learned-value top-1 reference selection stayed at 0.266--0.281. Top-20 CEM
reference recall was non-monotonic (0.148, 0.141, 0.070, 0.102, 0.188), and
every cell solved 0/2 episodes with zero valid sentences. MSE-only K=94 was
also unsuccessful. These are single-seed diagnostics, not a hierarchy claim.

## Falsifiable next test

Train three otherwise matched K=32 cells using prior-only, mixed, or random
primitive alternatives. The proposal-only change preserves the model, data,
seed, loss weights, macro K=16, and evaluation. Mixed/random coverage wins if
it improves held-out full-vocabulary reference top-5 and constrained-planning
top-20 recall without reducing pair accuracy below 0.80 or worsening drift and
collapse diagnostics. If prior-only is equal or better, retain it and wait for
the pending continuation-horizon evidence. If none improves executable validity,
stop proposal tuning and treat the learned value interface as the bottleneck.

## Validity and scale gates

All alternatives are tokens from the full vocabulary; there are no symbolic
feasibility labels, auxiliary language model, or candidate-privileged targets.
Report duplicate/unique candidate coverage before interpreting differences.
Require deterministic EMA targets, finite losses, healthy effective rank, and
matched sample counts. This is a one-seed screen capped at 7.5 GPU-hours; it
cannot justify scale-up.

## Steering-note effects

- The 2026-07-16 note required coordinate-correct reachability and non-symbolic
  primitive/macro advantages. It rules out symbolic candidates and keeps
  calibration, ordering, recall, and planning separate.
- The 2026-07-17 note explicitly prioritized the completed K-saturation audit
  and named prior-versus-mixed proposals as a conditional follow-up. The flat
  K result selects that follow-up, keeps macro K fixed, uses one GPU per job,
  and avoids paper-scale work. Its pending horizon matrix is not duplicated.

