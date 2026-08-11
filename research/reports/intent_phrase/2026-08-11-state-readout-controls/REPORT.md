# State-readout controls: is feasibility really absent from the pooled state?

_2026-08-11. Owner-prompted controls after the menu-free chance-floor result.
Checkpoint: stab-ldad-ema-s0-v1 (LDAD headline recipe, seed 0), frozen.
**All labels are oracle-derived — candidate-privileged diagnostic, not a
planning result.** Script: `scripts/probe_state_feasibility.py`; raw JSON in
the session scratchpad (`full.json`)._

## Question

The contrastive action prior (catalogue-softmax on the pooled state s_t with
detached candidate embeddings u(c)) ended at chance, and we claimed "the
pooled state lacks per-candidate prerequisite information". Yet LDAD
cycle-consistency reaches feasibility AUC .94 from the same (s_t, u(c)).
Which is it: (A) information absent from s_t, (B) present but the conjunctive
composition is hard for an imitation-trained head, or (C) mundane
undertraining/undercapacity?

## Setup

2000 train / 800 val problems (disjoint), frozen model, CPU (~9 min).
s_t = 256-d pooled state before step t; u(c) = 16-d frozen action-encoder
embedding; catalogue = all problem variables (mean 9.14).

## Control 1 — direct probes (oracle labels)

Resolvedness ("is variable v resolved at step t"; majority .781):

| probe on [s_t ; u(v)] | acc | AUC |
|---|---|---|
| logistic regression | .849 | .905 |
| MLP-128 | **.939** | **.982** |
| MLP-128, shuffled state | .780 | .722 |

Per-depth AUC (MLP-128): d0 .972, d1 .994, d2 .987, d3 .970, d4 .942, d5 .941.
One-hot-variable variant: logreg .900, MLP .910, shuffled .760.

Feasibility ("is candidate c executable now"; majority .613):

| probe on [s_t ; u(c)] | acc | AUC |
|---|---|---|
| logistic regression | .755 | .790 |
| MLP-128 | **.867** | **.935** |
| bilinear s^T W u | .854 | .925 |
| MLP-128, shuffled state | .731 | .743 |

Per-depth AUC (MLP-128): d0 .965, d1 .817, d2 .853, d3 .805, d4 .728, d5 .716.
One-hot-variable variant (parent names stripped): logreg .650, MLP **.755**,
bilinear .737, shuffled .538.

## Control 2 — catalogue-softmax heads at scale (imitation objective)

Chance E[ln M] = **2.188**; feasible-entropy floor E[ln #feasible] = **1.156**
(mean 3.54 feasible of 9.14). 8,913 train / 3,624 val steps.

| head | params | val CE | vs chance | top-1 | feas. AUC |
|---|---|---|---|---|---|
| (a) original prior head (h=256) | 71k | 1.716 | −0.47 | .286 | .823 |
| (b) wide MLP 512×3 | 666k | 1.702 | −0.49 | .282 | .806 |
| (c) bilinear + MLP hybrid | 408k | 1.864 | −0.32 | .283 | .830 |
| (d) cross-attn over sentence seq | 1.65M | **1.636** | −0.55 | **.334** | .737 |

## Verdict

- **(A) refuted.** Feasibility is decodable at AUC .935 from (s_t, u(c)) —
  matching LDAD's .94. Resolvedness at .982, near-uniform across depths.
  Both far above shuffled-state controls, so not a label-prior artifact.
  The earlier claim "the pooled state does not carry per-candidate
  prerequisite information" is wrong and has been corrected in the campaign
  log and interface report.
- **(C) partly implicated.** Trained offline on cached frozen features, the
  *identical deployed head* reaches CE 0.47 nats below chance — not at the
  floor. The in-trainer chance-level result reflects the training setting
  (detached auxiliary loss on a schedule tuned for the JEPA objective) at
  least as much as the head. Caveat: offline task replication, not a rerun
  of that cell (its chance floor was 2.57 vs 2.188 here — configs differ).
- **(B) is the real residual.** Capacity does not close the gap: all four
  heads sit within 0.23 nats of the 71k head and ≥0.48 nats above the
  feasible-entropy floor, while a *directly supervised* MLP of the same
  shape reaches .935 on identical inputs. Head (d), with 23x the parameters
  and the full pre-pooling sentence sequence, buys CE via trace preference
  while having the WORST feasibility AUC (.737). The imitation objective
  (next-action CE) rewards preference and only weakly rewards feasibility,
  so gradient descent never invests in the conjunctive parent-lookup. That
  the lookup is genuinely conjunctive: stripping parent names from the
  candidate embedding (one-hot variant) drops feasibility .935 → .755 while
  resolvedness barely moves (.982 → .910); per-depth AUC degrades
  monotonically from .965 (leaves) to ~.72 (depth 4-5).

## Consequence for the paper

LDAD's menu-free advantage is not privileged information — both routes see
the same (s_t, u). LDAD wins because its scoring rule (reconstruct the
candidate's own phrase from the predictor's imagined displacement) exposes
the state-action binding through the *trained dynamics*, whereas a fresh
head must rediscover it from imitation gradients that barely reward it.
Corrected claim: "state-only readouts trained by next-action imitation stay
near chance on feasibility; the same information is demonstrably present
(oracle probe AUC .935) and is recovered through the trained dynamics by
cycle-consistency." The .935 probe is the (oracle-supervised, hence not
itself deployable) ceiling for any future self-supervised feasibility head.

Limitations: single seed (0); 2000/800 problem subsample; trainer cell
replicated in task form, not rerun.
