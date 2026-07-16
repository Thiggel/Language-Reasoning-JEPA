# Cycle: hierarchy gradients and abstraction

Status: fixed-span evidence audited; semantic comparison blocked on one matched control

## Decision

Does adding fixed-span causal hierarchy change the shared token state toward
future-relevant abstractions, and does increasing the relative weight of the
longest-horizon loss improve that change without destabilizing prediction?

## Predeclared outcomes

- Continue to semantic phrase/sentence boundaries if L2/L3 increases held-out
  future, feasibility, value, or answer information relative to the
  capacity-matched flat objective while preserving state/action rank.
- Change loss balancing if encoder-gradient norms or cosines show low-level
  domination/conflict and duration-weighting improves the abstraction probes.
- Redirect target construction before scaling if hierarchy only improves
  position/token recovery, collapses, or fails the shuffled/flat health
  comparisons.

## Design

All cells instantiate the same `[4,16,64]` architecture so inactive-level
controls are parameter matched. Level weights select flat-only, L1, L2, or L3
training. The weighting screen holds the mean active hierarchy weight roughly
constant while changing the relative longest-level weight. Models use fresh
6k-example epochs, long 6--12-step traces, three exploratory epochs, and seed
0 with seed-1 replication for the key L2 comparison.

Primary evidence is levelwise token/future/answer probing plus encoder-gradient
norm and cosine diagnostics. Prediction drift, action rank, state variance,
and direct-versus-recursive endpoint error are health gates. These are
exploratory estimates, not paper-grade results.

A four-cell companion pilot compares a variable-duration phrase/sentence
hierarchy against (i) the same architecture with hierarchy weights zero and
(ii) random boundaries with the same number of segments per sentence. A
sentence-upweighted cell tests gradient competition. Phrase boundaries use
only rendered text markers (`is`, `=` and the sentence end), never symbolic
feasibility or graph state. Variable lower actions are encoded by a masked
bidirectional CLS Transformer.

## Important correction

The audit found that token-hierarchy VICReg had been applied to the EMA target
inside `no_grad`, producing no encoder gradient. The new round regularizes the
online causal states and regression-tests that encoder gradients are nonzero.
Historical token-hierarchy runs therefore did not actually test active VICReg.

## Scale gate

No 50M/100M run is justified until a 10M cell passes the abstraction and health
gates. Reachable transition-bank planning, goal-conditioned primitive
proposals, and value-aligned planning remain subsequent bounded cycles rather
than being mixed into this diagnostic.

## Interim results (2026-07-16 10:10 CEST)

The twelve fixed-span cells completed without runtime failure. A corrected
same-token comparison evaluates every seed-0 encoder at exactly the same token
positions. Relative to the capacity-matched flat encoder, linear CKA is
0.676--0.729, so hierarchy materially changes the shared coordinate space.
Remaining-work linear R2 changes from .573 (flat) to .687 (L1), .676 (L2
equal), .619/.601 (L2 sqrt/duration), .697 (L3 equal), and .497 (L3 duration).
Token-identity accuracy remains 1.0 in every cell. This is evidence for a
representation change and sometimes stronger progress information, but not
yet evidence that the shared token state discards lexical detail.

The earlier within-model endpoint probes are not used for this claim because
fixed-span endpoints differ systematically from all token positions.

A no-LM oracle-terminal CEM factorial is active on the L2 duration model. The
first completed flat, unconstrained, global-bank, conditional-bank,
conditional-prior, and adaptive-feedback cells all have zero solved episodes
and zero valid sentences over eight episodes. Conditional support improves
latent goal distance modestly, but one-token execution drift remains roughly
.61--.69. The current localized failure is therefore the primitive inverse /
conditional language-support interface, not only arbitrary off-bank macro
vectors. Reachability-crossed cells are still running.

## Terminal-summary audit and next decision (2026-07-16)

The newly terminal compact summaries were audited without opening raw logs or
sibling-project memory. Process state and scientific validity were treated
separately:

- all twelve fixed-span hierarchy cells completed with exit code 0;
- all twelve oracle-terminal CEM diagnostic cells completed with exit code 0;
- three semantic-boundary cells completed with exit code 0; and
- the semantic flat-objective cell timed out with exit code 124.

Every summary has an empty `metrics` object, no declared artifacts, and
`scientific_validity: not_assessed`. They therefore establish terminal process
state only. They do not validate, invalidate, or extend the interim scientific
metrics above. In particular, terminal completion of the reachability-crossed
oracle cells is not evidence of executable planning.

No token_igsm steering inbox existed at the supplied integration-repository
path. Consequently there were no unhandled steering notes to incorporate, and
no scientific choice was changed by a note. The supplied allocation snapshot
did change the operational choice: token_igsm has one guaranteed GPU and 15.0
project GPU-hours remaining, but only 0.9 GPU-hours remain globally. That rules
out seed expansion and admits at most one short recovery job.

### Falsifiable decision

Before interpreting the semantic-boundary pilot, recover the missing
information-matched flat-objective control at seed 0. After recovery, continue
semantic-boundary work only if the semantic cell improves the predeclared
future/progress or answer probe relative to **both** the random-matched-boundary
cell and the flat-objective cell, while state/action variance and effective rank
remain healthy. If it does not, keep the result as a negative pilot and redirect
boundary construction. If the recovery again fails to finish or produces
invalid/missing metrics, repair the measurement pipeline before further GPU
training. A single exploratory seed cannot trigger scale-up.

### Minimal recovery design

Run only `semantic-flat-objective-seed-0-recovery` with the existing semantic
architecture, data, seed, optimizer, three-epoch schedule, and rendered-text
boundaries, but set both higher-level loss weights to zero. The random-matched
and semantic cells are already terminal; rerunning them would exceed the global
headroom without resolving the immediate missing-control problem. The recovery
has a 50-minute one-GPU walltime (0.83 GPU-hours), writes a checkpoint plus
representation and gradient diagnostics, and does not submit automatically.

Primary comparisons after recovery are final-answer accuracy and
remaining-fraction R2 at token, phrase, and sentence levels. Health gates are
finite losses, non-collapsed state/action standard deviation and effective
rank, and nonzero encoder gradients for active objectives. Endpoint-token and
segment-length probes are diagnostics for lexical/length shortcuts, not proof
of abstraction. Oracle-terminal CEM remains a separate, privileged diagnostic.
