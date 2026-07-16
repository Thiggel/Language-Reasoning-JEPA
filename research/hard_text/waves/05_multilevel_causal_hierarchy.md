# Wave 05: multilevel causal token hierarchy

## Question

Can hierarchy become useful when primitive actions are tokens and higher
levels predict farther-away prefix states? Success is measured by end-to-end
generation, not merely high-level latent MSE.

## Implementation

- Level 0 consumes one observed token action per transition.
- Every transition model is a causal transformer over its state/action history.
- Higher actions preserve the complete ordered lower-action span through a
  concatenation/projection bottleneck; nested spans are exactly divisible.
- EMA targets, shifted recursive dense rollout at every level, terminal-state
  prediction, remaining-horizon value, conditional support, and high/low
  reachability are trained jointly.
- Checkpoint selection uses prediction, dense-rollout, reachability, and goal
  terms only, so calibration terms cannot hide poor dynamics.

## Stage-1 submitted screen

The screen uses long 10--18-step traces, 12k examples, five epochs, and seed 0.
It runs on three dedicated H100s while the easy-domain matrix continues on
other servers. Selected cells will be rerun at full scale and three seeds.

During the exploratory screen, planning uses only 8 episodes, beam 4, branch
2, and at most 8 executed macros; probes use 256 examples. These estimates are
diagnostics, not paper results. Only selected recipes receive the larger
budgets below.

| Factor | Cells |
|---|---|
| Control | capacity-matched flat token dynamics |
| Level-1 span | 4, 8, 12, 16 tokens at width 32 |
| Bottleneck | 8, 16, 32, 64 at span 8 |
| Dense recursive depth | 1, 4, 8 at span 8 |
| Interface loss | reachability on/off |
| Two levels | 8/24, 8/32, 8/40, 10/30 |

The sweep is staged rather than Cartesian. A hierarchy advances only if it
improves high-level prediction/reachability without collapsing action rank and
produces a planning gain over the same proposal LM.

## Planning evaluation

The auxiliary-LM proposal interface below was an initial engineering
diagnostic, not the intended primary experiment:

1. LM-only beam generation.
2. Flat JEPA scoring of identical proposals.
3. Discrete hierarchical scoring of encoded text spans.
4. Faithful top-down planning: CEM optimizes a high-level latent trajectory;
   its first state becomes a subgoal; CEM descends each macro level; token
   proposals retrieve and execute the first primitive span.

Reported CEM defaults to 1,000 candidates, 20 updates, 100 elites, and smoothed
distribution updates. Engineering smokes with smaller budgets are not faithful
results. Final diagnostics compare 900/1,000/3,000 candidates and 15/20/40
updates.

The primary oracle-goal diagnostic must not use an auxiliary LM. It optimizes
macro-actions with CEM, recursively turns the first predicted macro-state into
subgoals for lower levels, and finally uses categorical CEM over the token
vocabulary. The flat control uses the same categorical planner and primitive
model without macro subgoals. LM-proposal results are retained only to diagnose
inverse mapping/proposal coverage and are excluded from the main conclusion.

Support/reachability variants are staged as: unconstrained latent CEM; global
macro-code bank projection; state-nearest conditional code bank; conditional-
prior noise optimization; top-N low-level reachability reranking; and finally
ensemble disagreement for the selected architecture.

### Submitted fast oracle-CEM matrix (July 15, 2026)

The selected two-level model receives a strict 6 x 2 factorial: unconstrained
Gaussian CEM, learned conditional support energy, global codebook projection,
state-conditional codebook projection, full-covariance GMM energy, and a
state-conditional Gaussian prior, each with planning-time primitive
reachability refinement off/on. Macro CEM uses 256 candidates and 5 updates;
the nested token search uses a smaller exploratory budget. Additional controls
are flat categorical token CEM at horizons 8/32/96 and one- versus three-level
transfer checks for unconstrained, global-codebook, and conditional-prior
planning, again with reachability off/on.

The epistemic condition uses five independently initialized bootstrap macro
predictors trained against one frozen encoder, EMA target, action encoder, and
latent coordinate system. Variance across independently trained full JEPAs is
not used because arbitrary coordinate rotations would make it meaningless.

At every decision we separate proposal failure from ranking failure: reference
absent, reference available but rejected, invalid sentence, and valid but wrong
transition. Oracle terminal latents separate planning/model failure from goal-
head failure. We log value, goal, support, reachability, and CEM cost.

## Representation and failure probes

At token and every macro level, held-out linear probes measure token identity,
type, numeric value, sentence position, punctuation distance, remaining
horizon, and final answer. Diagnostics include state/action standard deviation
and effective rank, first/last-token recovery from macro codes, support AUROC,
direct high-level error, recursively composed low-level error, and goal error.

The desired abstraction signature is reduced local token/position information
with retained or improved answer/horizon information, coupled to better
planning. Oracle-only gains implicate the goal head; rejecting an available
reference implicates ranking/support; unreachable optimized states implicate
the top-down interface.

The submitted diagnostics additionally separate teacher-forced one-step error
from true recursive rollout-horizon drift, compare online and EMA coordinates,
and measure primitive-composed versus macro-predicted endpoints. Grouped
episode splits probe graph size, variable identity/value/operator/depth,
parent edges, query ancestry, resolved/feasible/unresolved-necessary sets,
progress, next operation, and the final answer for matched flat, one-, two-,
and three-level encoders.

An implementation audit found that the original controller-reachability target
discarded the causal transformer's state/action prefix when executing each
primitive chunk. Those existing checkpoints are therefore retained as an
explicit pre-fix diagnostic, not definitive evidence about reachability
training. The target now executes every chunk with its complete observed
prefix; an exact regression test covers non-initial macro windows. Fresh
`[8]`, `[8,32]`, and `[8,32,96]` checkpoints and a repeated 6 x 2 planner
factorial are queued as the corrected confirmation.

### Cross-level feedback MPC and phase-augmented dense prediction

The corrected `[8,32]` model is evaluated with fixed L2-boundary replanning,
L2 replanning after every executed 8-token L1 chunk, and adaptive L2
replanning when reached-versus-requested L1 waypoint error exceeds 0.3, 0.5,
or 0.7. These controllers are crossed with unconstrained, conditional-codebook,
and conditional-prior macro planning and reachability refinement off/on.
Off-boundary replanning restarts the upper causal predictor from the actual
encoded state rather than reusing a history ending before the correction.

The training screen addresses the resulting phase shift directly. Each L2
example samples phase 0, 8, 16, or 24 tokens and constructs a non-overlapping
causal 32-token trajectory from that phase. This trains every L1-boundary phase
without incorrectly treating overlapping actions as consecutive transitions.
Recursive dense supervision tests depths 1/2/4/8, high-only versus low-only
depth 4, and discounts 0.5/0.7/0.9/1.0. Every cell is evaluated with boundary,
L1-feedback, and adaptive MPC, each with reachability off/on.

Exploratory three-episode results show that phase augmentation is essential.
With the conditional codebook, the non-augmented boundary controller has mean
actual-to-goal distance 1.001. Phase augmentation alone reduces this to 0.847
at dense depth 1; depth 2 plus adaptive feedback and reachability reaches
0.589, a 41% reduction. Always-on feedback is less reliable than adaptive
feedback. Dense depth 2 is the current best trade-off: it removes the sharp
token rollout error increase beyond one step, while depths 4 and 8 do not
consistently improve planning. Discount changes are secondary; 0.5--1.0 all
produce competitive cells. Despite substantially better latent control, every
cell still has zero valid complete sentences, isolating primitive long-span
token realization as the remaining end-to-end bottleneck.

One-token execution MPC was then tested because the inverse audit recovered
96--98% of tokens when given the true next-state target. It does not solve
hierarchical generation. The selected phase/depth-2 model obtains goal
distance 0.814--0.833 under token MPC, versus 0.589 for its best 8-token
adaptive controller, and still produces no valid sentence. Primitive one-step
error remains moderate, but grammatical validity does not improve. Recursive execution error
is therefore not the sole bottleneck: a predicted macro waypoint is an
underdetermined token target, and unrestricted latent optimization can select
tokens that reduce model energy without remaining on conditional language
support. The next interface test must constrain primitive actions with a
state-conditioned action prior trained jointly in the same world model, or
with observed text-chunk support, and include pure-prior controls. An external
proposal LM remains excluded.

## Next decision

Select at most two Level-1 cells and one two-level cell. Run three seeds,
dense-depth/discount refinements, deterministic versus variational macro
encoding, and matched planner budgets. Add semantic phrase/sentence boundaries
only after fixed-span planning exceeds the flat control.

For the exploratory abstraction diagnostic, compare the shared encoder at
identical token positions across flat, `[8]`, `[8,32]`, and `[8,32,96]`
models. Probe lexical identity separately from future-relevant variables
(remaining horizon, resolved structure, final answer, and sentence-scale
progress); retaining token identity does not by itself refute abstraction.
