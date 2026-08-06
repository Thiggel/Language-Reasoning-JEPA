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
