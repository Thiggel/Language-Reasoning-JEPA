# Coherent endpoint Energy: horizon input is unnecessary; sparse prefix supervision rejected

_2026-08-06. Rounds `2026-08-06-intent-coherent-endpoint-energy-v1`,
`2026-08-06-intent-depth2-reference-evals-v1`,
`2026-08-06-intent-sparse-multidepth-energy-factorial-v3` (Grünau part).
One seed per new cell unless stated; 300 validation episodes; beam width 8;
root-balanced beam; depths 1,2,4,8,16; current feasible menu at depth 1,
symbolic future feasible tree (candidate-privileged) at depth > 1.
Snapshot `99e57f2`._

## Question

The validated endpoint-Energy recipe was internally incoherent: the Energy was
trained at horizons {1,2,4,8} but queried at evaluation depth 16 and at every
intermediate beam-pruning depth, and the 0.25 root pair-difference auxiliary
gave the same head a second semantics. Does removing all of that preserve the
depth-scaling result?

## New cells (all: logistic endpoint ranking weight 1, auxiliary 0, dense 0,
LR 3e-4, 300k examples, batch 32)

| Cell | Training horizons | Horizon input | D1 | D2 | D4 | D8 | D16 |
|---|---|---|---:|---:|---:|---:|---:|
| coherent-pow2 | {1,2,4,8,16} | yes | .133 | .463 | .797 | .807 | .810 |
| coherent-pow2-nohorizon | {1,2,4,8,16} | **no (zeroed)** | .113 | .450 | .787 | .797 | .807 |
| coherent-full16 | {1,...,16} | yes | .127 | .407 | .790 | .823 | .823 |

Reference (5 seeds, aux .25, horizons {1,2,4,8}; D2 added today):
strict .120 / **.461 ± .031** / .834 / .874 / .877.
Fixed-H4 reference at its untrained depth 2: **.290** (vs .910 at trained D4).

## Findings

1. **The horizon input contributes nothing.** The horizon-blind head (one
   shared Energy `E(root, endpoint, z_0)` for every depth) matches the
   horizon-conditioned twin at every depth within one-seed noise. The
   out-of-support-depth objection dissolves: there is no depth conditioning to
   be out of support of.
2. **Full integer-horizon support buys nothing** (full16 ≈ pow2), consistent
   with (1): the Energy generalizes across query depths because it never uses
   the depth.
3. **The scaling shape survives the fully coherent formulation** (monotone
   .11→.45→.79→.81 with depth; slack-2 reaches .99+ from D4). The mixed
   five-seed reference is ~.03–.07 higher at D8/16; whether that gap is the
   auxiliary, the horizon set, or one-seed noise is exactly what the running
   five-seed promotion will answer.
4. **Fixed-H4 is confirmed incoherent off-support** (.29 at D2), so the
   handbook's fixed-H4 alternative is rejected as the paper method.
5. **Sparse multidepth prefix supervision is rejected.** Every factorial cell
   (energy loss ∈ {MSE, rank, both} × geometry ∈ {none, straightening,
   projected straightening, monotonicity, projected monotonicity}, batch 16,
   prefix depths {0,1,2,4,8,16}) lands far below the plain terminal-endpoint
   recipe — best cell dense-none/both D16 = .55, most cells .1–.3, many with
   non-monotone depth curves. Straightening and monotonicity (raw or
   projected) show no consistent benefit in combination with ranking in this
   formulation. Lise dense-geometry cells are still running but cannot
   plausibly reverse a 30-point gap.

## Decision

Paper-method candidate: **horizon-blind endpoint-ranking Energy** —
`E(z_root, z_endpoint, z_0)`, logistic pairwise ranking of imagined endpoints
labeled by EMA-latent distance of their true outcomes to the encoded solved
state; mixed rollout horizons {1,2,4,8,16}; no root auxiliary; no dense loss;
no horizon conditioning. Five-seed promotion of this cell (s1–s4) and of its
horizon-conditioned twin (s1 running, s2–s4 queued) launched tonight.

## Five-seed promotion (added same day)

Horizon-blind endpoint ranking, five seeds (s0-s4):

| Depth | Strict | Slack-2 |
|---:|---:|---:|
| 1 | .123 ± .009 | .556 ± .014 |
| 2 | .438 ± .041 | .869 ± .010 |
| 4 | .777 ± .023 | .985 ± .005 |
| 8 | .800 ± .022 | .994 ± .003 |
| 16 | .801 ± .024 | .995 ± .004 |

Horizon-conditioned twin (2 seeds): .135/.447/.780/.820/.818 — overlapping;
the horizon input is confirmed unnecessary.

The remaining ~.05-.07 deep-depth gap to the old mixed reference
(.834/.874/.877 at D4/8/16) therefore tracks the 0.25 root pair-difference
auxiliary, not the horizon input. Note that under a horizon-blind head the
auxiliary is no longer semantically incoherent: there is only one Energy
semantics for it to distill into. The decisive ablation
(`mix_pow2_16_aux025_nohorizon`, five seeds) is running; if it recovers the
reference level, the frozen paper recipe is horizon-blind ranking + 0.25
auxiliary, all coherent.

## Caveats

- Depth > 1 remains candidate-privileged (symbolic future feasible menus).
- One seed for the new cells until the promotion completes.
- The 0.25-auxiliary variant under the pow2 horizon set has not been run; if
  the five-seed gap at D8/16 persists, that is the next single ablation.

## Recipe freeze (2026-08-06 night)

Horizon-set ablation under the horizon-blind head (0.25 auxiliary in both):

| Training horizons | D1 | D2 | D4 | D8 | D16 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| {1,2,4,8} | .127 | .443 | .827 | **.870** | **.875** | 2 |
| {1,2,4,8,16} | .121 | .422 | .808 | .832 | .833 | 3 |

Training-budget dilution by H16 rollouts explains the earlier deep-depth gap;
the horizon-blind Energy generalizes to query depth 16 without ever training
on H16 rollouts.

**Frozen paper recipe**: horizon-blind endpoint Energy
`E(root, imagined endpoint, z_0)` (no horizon input), logistic pairwise
ranking of imagined endpoints labeled by EMA-latent distance of their true
outcomes to the encoded solved state, training rollout horizons {1,2,4,8},
0.25 root pair-difference auxiliary (coherent: one shared Energy semantics),
no dense rollout loss. Matches the historical mixed-horizon headline
(.834/.874/.877 at D4/8/16) with every incoherence removed. Five-seed
promotion of the frozen recipe launched (s2, s3 running; s4 pending a free
GPU).

Same-protocol competitor energies (one seed each): TD-Q (SARSA, amortized)
.190/.210/.240/.223/.220 — myopically stronger at D1, cannot exploit search;
expectile goal-value .057-.187 — near random. See
`../2026-08-06-competitor-energy-baselines/REPORT.md`.

## Final five-seed frozen-recipe headline (2026-08-07)

| Depth | Strict | Slack-2 |
|---:|---:|---:|
| 1 | .126 ± .005 | .541 ± .023 |
| 2 | .450 ± .031 | .863 ± .008 |
| 4 | .837 ± .019 | .979 ± .008 |
| 8 | .879 ± .027 | .991 ± .004 |
| 16 | .884 ± .023 | .995 ± .004 |

References (same episodes): random .053, first-feasible .247, oracle 1.000.
The frozen coherent recipe slightly exceeds the superseded horizon-conditioned
headline at every depth >= 4.

## Frozen-recipe variants (2026-08-07, one seed each)

| Variant | D1 | D2 | D4 | D8 | D16 |
|---|---:|---:|---:|---:|---:|
| Frozen recipe (5-seed reference) | .126 | .450 | .837 | .879 | .884 |
| + expectile-TD shaping auxiliary | .150 | .323 | .847 | .863 | .863 |
| Non-residual (direct) MLP prediction | .120 | .410 | .813 | .850 | .860 |

- **TD shaping does not improve the frozen recipe**: deep depths within one
  seed SD, and a clear D2 regression (.323 vs .450). TD-as-ingredient is
  retained as a documented negative ablation, consistent with the pure TD
  baselines' failure to exploit search.
- **Residual prediction is roughly neutral for the MLP predictor** (slightly
  below reference at every depth, within ~1 SD). Combined with the earlier
  causal-matrix finding that the transformer predictor prefers direct
  prediction, the honest claim is: the residual choice is second-order; the
  predictor class (MLP vs transformer) is the first-order choice.
