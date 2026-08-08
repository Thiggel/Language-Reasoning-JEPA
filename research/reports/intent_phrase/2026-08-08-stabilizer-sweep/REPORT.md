# Stabilizer sweep, stage 1: SIGReg upgrades the recipe

_2026-08-08. Round `2026-08-08-intent-stabilizer-sweep-v1`, snapshot
`6cf9424`. Joint-training screening cells, identical settings to the
frozen recipe except the named stabilizer change; 300 val episodes,
root-balanced beam B=8, slack-curve evaluator, strict success D1..D16._

Owner directive: sweep anti-collapse variants (no stopgrad/EMA, LDAD
per Delta-JEPA, {VICReg, SIGReg, VISReg} x target modes).

| cell | stabilizers | strict D1/D2/D4/D8/D16 |
|---|---|---|
| reference (5 seeds, 2026-08-07) | EMA + stopgrad + VICReg | .126/.450/.837/.879/.884 |
| **sigreg (5 SEEDS)** | EMA + stopgrad + SIGReg | **.143/.434/.888/.959/.959** (sd .011-.018) |
| online_nosg (1 seed) | VICReg only (no EMA, no stopgrad) | .093/.317/.670/.727/.727 |
| ldad_only (1 seed) | LDAD only (no EMA/stopgrad/VICReg) | .080/.223/.637/.723/.720 |

Findings:
1. **SIGReg replaces VICReg and gains ~+.07 at D4-D16, replicated over
   five seeds** (D16 .959+-.011 vs .884+-.023). Everything else equal.
   This is the new headline candidate.
2. Delta-JEPA's stability claim HOLDS: both no-EMA/no-stopgrad cells
   train without collapse (LDAD alone suffices as the anti-collapse
   device). But both lose ~.16 at depth vs the reference — EMA+stopgrad
   contribute planning performance, not just stability.
3. Pending stage 2: VISReg port (arXiv 2606.02572), LDAD+EMA+stopgrad
   combination, VICReg/SIGReg/VISReg combinations, and frozen-protocol
   replication (distill the Energy head from a frozen SIGReg backbone)
   to remove the EMA-label confound from the comparison.

## Update: LDAD five-seed result — new headline recipe

LDAD added to the unchanged recipe (EMA+stopgrad+VICReg kept), five
seeds, strict D1..D16: **.202+-.049 / .729+-.026 / .885+-.026 /
.965+-.015 / .975+-.012** — dominates the 2026-08-07 recipe at every
depth. The D2 gain (+.28) removes the recipe's known shallow-depth
weakness, matching theory prediction T4 (the best-of-R rollout label is
least reliable at horizon 1; LDAD's latent action-displacement decoding
supervises exactly the one-step geometry the label cannot).

Frozen-protocol check (1 seed): an Energy head distilled from the frozen
LDAD backbone reaches .183/.763/.887/.940/.940 — the shallow-depth gain
persists, so it lives in the backbone geometry, not the joint training.

Composition: SIGReg+LDAD without VICReg does NOT compose
(.130/.403/.783/.840/.843, 1 seed). VICReg+SIGReg+LDAD three-way cell
pending. Current recommendation: promote LDAD+EMA+stopgrad+VICReg to the
paper headline; keep SIGReg-for-VICReg as the deep-depth-only runner-up
row in the sweep table.

Three-way VICReg+SIGReg+LDAD (1 seed): .120/.417/.777/.860/.867 — also
fails; SIGReg actively interferes with the LDAD gain regardless of
VICReg. Final sweep verdict: **LDAD+EMA+stopgrad+VICReg is the recipe**;
SIGReg is a deep-depth-only alternative that must not be mixed with LDAD.
