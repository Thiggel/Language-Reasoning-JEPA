# Intent-phrase J3 weekend recipe search

## Decision

Find a healthy, deployable recipe by testing one bounded mechanism at a time:
learning rate, width, exact dense recursive supervision, GAR breadth plus
advantage regression, then executable distinct-state hierarchy.  Flat strict
planning success is primary; slack-2 success, collapse/rank, action scale,
teacher ordering, and value calibration are validity diagnostics.

## Live evidence (2026-07-18 11:40 CEST)

- The complete 5e-4 proxy reached `.565` strict and `.820` slack-2.  Direct
  checkpoint inspection showed that 1e-3, 1.4e-3, and 2e-3 completed all ten
  training epochs before timing out in evaluation; only 7e-4 stopped after
  epoch 8.  Recovered planning gives 1e-3 `.675/.905` and 1.4e-3
  `.580/.815`, bracketing a clear seed-0 optimum near 1e-3.  The independent
  current-code 1e-3 anchor completed at `.660/.885`.  The 2e-3 validation loss
  is worse and its action-code scale is inflated, so additional local LR
  refinement no longer changes the decision.
- Width 384 and 512, dense depths 2/4/8, GAR K=8 rank-only, and two distinct
  hierarchy representation pilots are active.  Interpret only terminal
  planning artifacts; unequal throughput makes current step counts unmatched.
- GAR regression failures were invalid: masking after arithmetic allowed
  `0 * inf` to create NaN gradients.  The fixed loss masks before subtraction;
  full tests and a finite training smoke passed.  Fixed K=2/K=8 recoveries are
  active or queued.
- Distinct hierarchy representations remain non-collapsed in interim audits,
  but prior planning mixed low and high coordinates and discarded causal
  histories.  Commit `459af8f` lifts complete candidate paths through the EMA
  high encoder, retains primitive and macro histories, replans only on K-step
  boundaries, and iteratively refines primitive actions between boundaries.
  The full suite passes (101 tests).

## Active executable hierarchy test

Three exact-snapshot jobs are accepted on Lise/Grete: value weights 0.25 and
1.0 at LR 1e-3, plus a 7e-4 LR cross-check.  They use direct remaining-step
value labels (symbolic training supervision), dense four-jump macro prediction,
and compare low refinement horizons 1 versus 2.  Promotion requires improved
hierarchical planning without degrading the checkpoint's information-matched
flat planning or collapsing either state space.

## Next gates

1. Confirm the bracketed 1e-3 anchor on seeds 1 and 2 (accepted on Lise and
   Grete); use the three-seed distribution for all method promotion decisions.
2. Promote only the best healthy width; do not extend the local LR sweep unless
   seed confirmation contradicts the current 1e-3 bracket.
3. Compare dense and GAR cells against the matched flat anchor, then refine
   depth/weight or K/regression weight only inside viable regions.
4. Sweep macro bottleneck 4/8/16 only if the executable d8 pilot beats its own
   flat evaluation and shows calibrated high values.
