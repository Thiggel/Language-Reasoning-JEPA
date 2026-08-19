# Representation analysis battery: consequence geometry, probes, embedding maps

_2026-08-20. Code: `scripts/analysis/{consequence_geometry,probe_battery,embedding_maps}.py`
(+ `adapters.py`, `pairs.py`, `common.py`, `summarize_battery.py`), tests in
`tests/test_analysis_battery.py` (21 tests). Raw numbers and the full pair
lists are in `data/`; figures and their 2-D coordinates in `figures/`._

**Evidence labels.** Every "same consequence / different consequence" label and
every probe target in this report is an **oracle fact about the environment**.
Labels are used only to *group* pairs and to *train probes*; they are never a
model input. Nothing here is a planning result. The steps-to-go probe is a
**diagnostic only** — per `CLAUDE.md` it must never become a system component.

---

## 1. What we were trying to find out

The 2026-08-12 LM state-readout report killed the claim "JEPA states carry
prerequisite structure that LM states lack". The replacement claim under test
here is **consequence-based geometry**: the JEPA's predictor target should make
states and actions organise by *what they do*, not by *how they are worded* —
actions with identical consequences collapse together, actions with different
consequences separate — which is what a latent planner needs.

This battery tests that claim and, where it fails, says so.

## 2. How it is measured (plain English)

We build pairs of phrases using **the environments' own renderers**; no new toy
text is invented, only the arguments handed to those renderers change.

* **Paraphrase pairs** — same consequence, different words. *Stylized iGSM*:
  operand commutation, "derive X from A plus B" vs "derive X from B plus A"
  (addition and multiplication commute, so X gets the identical value).
  *ProofWriter*: the same rule application with its premises listed in the other
  order — premises are a set, so the derived fact is bit-identical.
* **Negation pairs** — different consequence, **one-token** change. *ProofWriter*:
  flip one premise's polarity, "Anne is white." -> "Anne is not white.", which
  denotes a different fact and makes the rule inapplicable. This is the real
  negation test. *Stylized iGSM has no negation*; the nearest minimal-edit
  contrast is an operator swap ("plus" -> "times"), reported as `operator_flip`
  and never called negation.
* **Control pairs** — different consequence, comparable surface change: a
  different operand, a different target variable, another legal rule application.

We then measure the distance between the two frozen representations (cosine, raw
L2, and L2 after length normalisation) and report a separation ratio and an AUC
— the probability that a random different-consequence pair sits farther apart
than a random same-consequence pair (.5 = no signal).

**Faithful iGSM has no action paraphrase at all**, and none was invented: its
intent phrase is `Define <name> .`, naming only the target, with no operator and
no operands. There consequence geometry is measured on the **state** side:
doing two independent steps A-then-B vs B-then-A yields different text but an
identical set of resolved quantities, while A-then-B vs A-then-C yields a
different set. Both orders are trajectories the environment itself generates.

### Two controls that decide what the numbers mean

1. **Surface baseline.** Every pair also carries its token-level edit distance.
   In stylized iGSM the paraphrase pairs are *farther apart in text* (3.81
   tokens) than the different-consequence pairs (1.60), so a purely
   surface-driven encoder scores **AUC .028** — the opposite direction. Any
   score above .5 must come from something other than the words.
2. **Random-init encoder.** The identical architecture with fresh weights. This
   is the "collusion" control, and in this battery it is decisive: several
   headline-looking numbers survive it, and several do not.

---

## 3. Results

### 3.1 ProofWriter — negation separation (the real negation domain)

Backbone-matched triple, all trained on the same compiled corpus. 345 contexts,
650 pairs. Action-in-context representation, cosine distance.

| model | negation pair | paraphrase pair | different legal action | AUC same-vs-diff |
|---|---|---|---|---|
| JEPA (pw-ldad-lr3e-4-s0) | .1284 | .0060 | .6170 | .984 |
| token LM (pw-tok-lm-lr3e-4-s0) | .0012 | .0531 | .0500 | **.170** |
| sentence LM (pw-sent-lm-lr3e-4-s0) | .1856 | .0004 | .3143 | 1.000 |
| JEPA, random init | .0214 | .0000 | .2639 | 1.000 |

Raw distances are not comparable across encoders (a nearly-collapsed encoder has
tiny distances everywhere, which inflates every ratio), so the load-bearing
table divides each family by the **same model's** distance for a wholly
different legal action:

| model | negation salience | paraphrase leakage |
|---|---|---|
| JEPA | **.208** | .0098 |
| token LM | .025 | **1.062** |
| sentence LM | .590 | .0013 |
| JEPA, random init | .081 | .0000 |

Reading it:

* **The trained JEPA separates negation 2.6x better than its own random-init
  control** (.208 vs .081) while keeping paraphrases collapsed (.0098). That
  margin is what JEPA training adds.
* **The token LM is anti-consequence** (AUC .170). A one-token meaning flip moves
  it only 2.5% as far as a completely different action, while a premise
  *reordering* — which changes nothing about the world — moves it **farther than
  a completely different action** (leakage 1.06). At matched positions the token
  LM's action geometry tracks word order, not consequences.
* **The sentence LM is the strongest on this metric** (.590), better than the
  JEPA. Consequence geometry is *not* JEPA-exclusive.

In the JEPA's **predicted next state** — the closest thing to "what the model
thinks will happen", and a space only an action-conditioned predictor has —
negation salience is **.364 trained vs .133 random-init** (2.7x), paraphrase
leakage .012 vs .000.

Surface reference for this domain: paraphrase pairs differ by 3.49 tokens,
different-consequence pairs by 3.93 (AUC .310) — surface again points the wrong
way, so the JEPA and sentence-LM numbers are not a surface artefact.

### 3.2 Stylized iGSM — where the raw metric fails its own control

200 contexts, 2404 pairs. **Raw AUC is 1.000 for the JEPA, the sentence LM, the
random-init JEPA and the random-init token LM alike.** Paraphrase collapse is
architectural here: commuted operands land at ~0 distance in every
chunk-pooling encoder, trained or not. **The raw stylized AUC is therefore not
evidence of learned consequence geometry, and we do not report it as such.**

Only the scale-normalized view separates the models (each family divided by the
same model's `different_operand` distance — a different-consequence pair with a
comparable surface change):

| model | paraphrase (commuted operands) | operator flip | different target |
|---|---|---|---|
| JEPA (stab-ldad-ema-s0) | .0136 | **5.06** | **60.3** |
| token LM | .692 | 74.7 | .156 |
| sentence LM | .0009 | .487 | .988 |
| JEPA, random init | .0000 | .582 | .948 |
| token LM, random init | .0008 | .425 | .989 |

The trained JEPA is the only model whose action geometry is **graded by how much
the consequence changes**: an operator flip moves it 5x and a different target
60x a plain operand swap, while paraphrases stay 70x tighter than that same
operand swap. The random-init JEPA, the sentence LM and the random-init token LM
all place operator flips and different targets at roughly 0.4-1.0 of the operand
distance — they respond to surface magnitude, not to consequence magnitude.
The token LM again barely collapses paraphrases at all (.692).

**State order-invariance on stylized is a negative result for the JEPA:**

| model | AUC (same-order vs different-consequence) |
|---|---|
| JEPA | .538 |
| token LM | .685 |
| sentence LM | .777 |
| JEPA, random init | .551 |
| token LM, random init | 1.000 — **degenerate**, both distances ~0 (.0000/.0003) |

The stylized JEPA's pooled state is at chance for order-invariance, below both
LMs. This is the pooled-state weakness the 08-12 report already flagged, showing
up again in a different instrument.

### 3.3 Faithful iGSM, flat JEPA — the backbone-matched before/after

This is the paper's central representation claim, and it is the cleanest result
in the battery. The FlatIntentJEPA encoder is *literally initialised from* the
token LM, so we can measure the **same weights** before and after JEPA training.
72 order-commutation cases; AUC is scale-free.

| arm | state order-invariance AUC |
|---|---|
| flat JEPA after training (`flat-lminit-ecf16-s0`, epoch 0 ckpt) | **.711** |
| flat JEPA at LM-init, **before** JEPA training | .568 |
| token LM it was initialised from (`tok-lm-med-fullsol-big-lr1e3-s0`) | .568 |
| flat JEPA, random init | .516 (chance) |

Sanity check that the arm is real: the `lm_init_untrained` row reproduces the
token-LM row to three decimals (.568), exactly as it must, since that arm *is*
the token LM's encoder; and the run log confirms
`FlatIntentJEPA: encoder initialized from .../tok-lm-med-fullsol-big-lr1e3-s0/model/best.pt`.

So on identical weights, JEPA training moves consequence-organisation of states
from **.568 to .711**, with a random-init floor of .516. Absolute distances also
shrink ~20x (.307 -> .015), i.e. JEPA training compresses the state manifold —
but the AUC is a rank statistic, so the gain is not a scale artefact.

Faithful action-side paraphrase collapse is **not measured, because the domain
has no paraphrase** (see §2); this is recorded in the JSON as an explicit note
rather than filled with invented text.

### 3.4 Probe battery — and the random-init control that reframes it

Stylized iGSM, 800 train / 350 val problems, MLP-128 probes, oracle labels.

| model | resolved (AUC) | feasible (AUC) | steps-to-go (R2, **diagnostic**) | step value (R2) | operator from displacement (acc) |
|---|---|---|---|---|---|
| JEPA | **.969** | **.918** | .297 | -.043 | **.995** |
| token LM | .906 | .883 | .325 | -.008 | .924 |
| sentence LM | .923 | .898 | .347 | -.174 | .891 |
| **JEPA, random init** | **.912** | **.878** | .323 | -.040 | **.960** |
| **token LM, random init** | **.899** | **.867** | .132 | -.163 | **.972** |

The trained-vs-random gap is the whole story: a **random-init encoder already
reaches .912 / .878**. The trained JEPA's entire margin over its own random
control is **+.057 resolvedness and +.040 feasibility**, and the trained LMs sit
*below* the random-init JEPA on both. Operator-from-displacement looks
spectacular at .995 but the random-init controls reach .960 and .972, so almost
all of it is architectural. The step's computed value is **not decodable at all**
(R2 <= 0 everywhere): modular arithmetic results are not linearly present in any
of these representations.

Per-depth feasibility/resolvedness reproduces the 08-12 finding exactly:

| model | resolved d0 | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|---|
| JEPA | .951 | .988 | .977 | .939 | .918 | **.602** |
| token LM | .846 | .953 | .946 | .958 | .984 | **1.000** |
| sentence LM | .880 | .958 | .944 | .916 | .932 | 1.000 |

The JEPA's pooled state **decays with dependency depth** (.951 -> .602) while
both LMs hold or improve. That is an honest JEPA weakness and it replicates.

### 3.5 Embedding maps

`figures/` holds t-SNE and UMAP of action and state vectors for the stylized
JEPA, token LM, sentence LM and the random-init JEPA, coloured by operator type,
dependency depth, feasible/infeasible, and paraphrase-pair membership
(commuted-operand partners joined by a line). PNG + PDF, with the 2-D
coordinates saved alongside as `coords_*.npz` so figures are reproducible
without re-embedding. The paraphrase-membership panels are the visual form of
§3.2: partners sit on top of each other in every chunk-pooling encoder including
the random-init one, which is precisely why the quantitative claim rests on the
normalized table and not on the picture.

---

## 4. What this does NOT show

* **It does not show that JEPA probes beat LM probes.** They do not. Feasibility
  and resolvedness are read out at .88-.92 from every family, and the 08-12
  conclusion stands unchanged.
* **It does not show that the probe numbers reflect training.** A random-init
  encoder reaches .912/.878. Any probe claim in the paper must be quoted against
  its random-init control or it is uninterpretable.
* **The stylized paraphrase-collapse AUC of 1.000 is not evidence of anything
  learned** — the random-init encoder scores 1.000 too. Only the scale-normalized
  gradedness (§3.2) survives that control, and it is a single-seed result.
* **Consequence geometry is not JEPA-exclusive.** On ProofWriter the sentence LM
  separates negation *better* than the JEPA (.590 vs .208 normalized).
* **The stylized JEPA fails the state order-invariance test** (.538, at chance,
  below both LMs). The positive order-invariance result is only on faithful with
  the flat backbone.
* **The flat-JEPA before/after result is one seed, one checkpoint, at epoch 0**
  of a still-running run, on 72 commutation cases, measured on `last.pt` because
  no `best.pt` existed yet. It needs seeds and a finished run before it is a
  paper number.
* **No planning claim is licensed by any of this.** Consequence geometry is a
  property of frozen representations; whether it converts into closed-loop
  planning is measured by the planning tables, and the 2026-08-19 oracle-free
  depth re-measure and anti-feasible energy findings still stand.
* **Nothing here measures the energy head or LDAD decodability**; those remain
  separate instruments.
* **Faithful action paraphrase collapse is unmeasured** (the domain has no
  paraphrase), so the faithful column of the paraphrase story is empty by
  construction, not by omission.

## 5. Limitations

Single seed per source throughout. Training-campaign confounds are uncontrolled
(the LMs saw far more data than the stylized JEPA; the flat JEPA is at epoch 0).
Widths differ across families (JEPA 256, token LM 288, flat 768). ProofWriter
premise-reorder pairs are only 75 of 650. The stylized `different_target`
control changes the most salient token in the phrase, so its 60x normalized
distance may partly reflect token salience rather than consequence magnitude.
The order-commutation test uses 72-151 cases per domain.

## 6. Pending

`2026-08-19-flat-jepa-v1` cells other than `flat-lminit-ecf16-s0` (`flat-lminit-s0`,
`-lr3e5`, `-nocfrank`, `-scratch`, `-noprior`, `-ecf1`, `-ecf64`, seeds s1/s2)
are still training; the battery already accepts them
(`--model flat_jepa=<ckpt> --flat-lm-init <ckpt>=<lm_ckpt>`). The
`nocfrank` ablation against `ecf16` is the natural next run, since it would show
whether the energy counterfactual-ranking term is what produces the .568 -> .711
move.
