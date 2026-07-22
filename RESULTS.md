# Results — experimental record (updated 2026-07-14)

Setup: iGSM-style synthetic reasoning (mod-23 arithmetic DAGs, 6–12
quantities, 3–9 necessary steps, distractor variables). All models ~9M
params, usually 20 epochs (V100/RTX6000, ~40–70 min per run). **Protocol
correction:** historical artifacts used a deterministic per-index corpus that
was reused each epoch; they were not fresh-data-per-epoch runs. The new
`FreshEpochSampler` supplies deterministic, disjoint epoch-offset samples, but
headline results have not yet been rerun with it. The historical 10k/30k/100k
comparison also changes optimizer-step count with corpus size and is
exploratory, not a controlled diversity ablation.
Planning success = query resolved within the **strict optimal step budget**
(every distractor pick is fatal at slack 0); planner sees only the action
interface, all consequence evaluation is latent. Baselines: random feasible
policy, symbolic oracle (100%). Full tables: `runs/report.md`; per-run
probes with random-encoder controls: `runs/*/probe_results.csv`;
latent-geometry plots: `runs/*/geometry.png`.

## Planning success (discourse track)

| run | @optimal, look-1 | @optimal, look-2 | @slack 2 | distractor rate |
|---|---|---|---|---|
| random policy | 0.055 | — | 0.405 | 0.43 |
| disc_no_delta | 0.150 | — | 0.510 | 0.40 |
| disc_base | 0.350 | — | 0.750 | 0.28 |
| disc_chunkpred | 0.620 | 0.755 | 0.880 | 0.16 |
| disc_combo | 0.635 | 0.760 | 0.890 | 0.17 |
| disc_valgrad | 0.665 | 0.835 | 0.920 | 0.07 (look-2) |
| disc_mono_hi | 0.655 | 0.735 | 0.935 | 0.15 |
| disc_rank_k2 | 0.910 | 0.850* | 0.985 | 0.04 |
| disc_champion (rank+mono) | 0.905 | — | 0.990 | 0.04 |
| disc_rank_cal (+cost ranking) | 0.905 | **0.945** | 0.990 | — |
| disc_champion_cal | 0.910 | 0.945 | **0.995** | — |
| **disc_mono_distill_rank** (no scalar labels!) | **0.935** | — | 0.990 | — |

*Deeper search *hurts* plain-ranking models (finding 12); CostRanking
fixes it (finding 15). Seed spread: combo 0.60–0.695, mono_hi
0.57–0.655, rank_k2 0.885–0.910, rank_cal 0.900–0.910, champion
0.87–0.905 @strict; slack-2 numbers stable within ±0.03. 1000-episode
confirmations: rank_k2 0.917, champion 0.913, rank_cal 0.914 (look-2
0.946), mono_hi 0.706, combo 0.653, base 0.391.

Edit track (value-head energy): edit_base 0.115 → **edit_valgrad 0.425**
(random 0.06). See finding 8 for the raw-geometry energy, which changes
this picture.

### Raw-geometry (goal-distance) planning — the second cycle

Energy = LN-L1 distance of the predicted latent to the encoded solved
terminal state (LeWM-style), **no value head at plan time**:

| run | straighten λ | @optimal | @slack 2 | value-energy @slack 2 |
|---|---|---|---|---|
| disc_combo | 0 | 0.150 | 0.520 | 0.890 |
| disc_straight_lo | 0.02 | 0.220 | 0.725 | 0.870 |
| disc_straight_mid | 0.05 | 0.365 | **0.845** | 0.730 |
| disc_straight | 0.10 | **0.425** | 0.795 | 0.645 |
| disc_mono (hinge only) | — | 0.165 | 0.590 | 0.875 |
| disc_mono_hi (w2, m.05) | — | 0.255 | 0.680 | **0.935** |
| disc_mono_novalue (no V) | — | 0.200 | 0.650 | 0.535* |
| disc_value_only | — | 0.060 | — | 0.430 |
| disc_proj_straight (π(s), 0.1) | — | 0.335 | 0.650 | 0.880 |
| disc_proj_geo (π(s), both) | — | 0.380 | 0.650 | 0.895 |
| disc_champion (rank+mono) | — | **0.655** | — | **0.990** |

The learned projection π(s) softens the trade-off (value planning fully
preserved at 0.88–0.895 with decent geometry 0.335–0.38) but direct
straightening still wins on pure geometry; disc_champion gets the best of
both without any curvature loss.

*disc_mono_novalue's value head is untrained; its 0.535 is the value
column's floor, not a real planner.

## The findings

1. **Latent planning over language works.** Searching tiny action codes
   with F(s, a) rollouts scored by a remaining-steps energy solves 84% of
   problems with zero wasted steps (look-2 valgrad), vs 5.5% random.
   Deeper search monotonically helps (66→84%), the signature of a
   trustworthy world model + energy.

2. **Encoder–predictor collusion is real and measurable.** With pure
   state-prediction JEPA, linear decodability of computed values from
   states *fell* during training (0.75→0.42; random-init encoder control:
   0.87), and the answer was barely present in the terminal state (0.26).
   The bottleneck kept "what the discourse is doing" (ops, structure,
   progress) and discarded "what it established" (values) — exactly the
   quantifier-collapse failure mode anticipated in the design discussion.

3. **Frozen-anchor chunk prediction cures it, reconstruction-free.**
   Predicting the *frozen random-init* chunk-encoder embedding of the next
   step from (s_t, a_t) (VL-JEPA-style continuous targets that cannot
   drift) restored value decodability (state 0.81–0.89, answer-from-final
   0.98) and produced genuine latent arithmetic: the computed value is
   decodable from *predicted* latents at 0.31 vs 0.15 random control —
   the predictor computes outcomes it never observes. Planning: 35→62%.

4. **Hybrid displacement decoding helps the early baseline, but is not a
   faithful Delta-JEPA replication.** Removing it halves planning (35→15%)
   and pushes chosen-action distractor rates toward random. However, this
   repository decodes a symbolic operation class plus an EMA intent embedding;
   Delta-JEPA decodes the observed raw continuous action end-to-end. Later
   counterfactual preference learning can largely replace the hybrid loss.

5. **Goal energy quality is the binding constraint.** Letting the value
   gradient shape the encoder (value_detach=false) was the single
   strongest planning lever on both tracks (discourse 35→66%; edits
   11→43%, helped by more vandal-edit negatives). The combo run shows you
   can have collusion-free representations *and* a sharp goal energy
   (value MAE 0.38, best) in one model.

6. **Temporal straightening makes the latent metric a progress metric**
   (arXiv:2603.12231, curvature loss 1−cos(v_t, v_{t+1}) on online
   states). Raw goal-distance planning — which fails on the unregularized
   space (0.10–0.18 @optimal) because the metric is not monotone in
   progress-to-goal — reaches 0.425 @optimal / 0.845 @slack 2 with a clean
   dose-response over λ ∈ {0, .02, .05, .1}: mean velocity cosine −0.25 →
   +0.67, corr(latent goal distance, true remaining steps) 0.73 → 0.86.

7. **But one latent space cannot currently serve both energies.** As λ
   grows, value-head planning falls (0.89 → 0.645 @slack 2) and value
   decodability from states drops 0.89 → 0.43 — straightening trades
   content for geometry. λ=0.05 is the raw-geometry sweet spot; the
   goal-monotonicity hinge is the content-preserving alternative: it sets
   the overall value-planning record (disc_mono_hi: 0.935 @slack 2) while
   only mildly improving raw geometry (0.255/0.68). A geometry-only model
   with **no value head at all** (disc_mono_novalue) still plans at
   0.20/0.65 vs 0.06/0.095 random; conversely disc_value_only's geometry
   plans exactly at random (0.06) — planning-relevant geometry comes from
   the JEPA objectives, not from value supervision.

8. **Edit-track plot twist: the geometry was never the problem.** The raw
   buffer-encoder goal distance beats every learned edit value head:
   edit_base 0.425/0.45 vs its value head's 0.115; edit_anchor + raw
   geometry is the best edit planner (0.47). The geometry regularizers add
   nothing here (goal_dist_corr was already ≈0.75) — the edit track's
   published weakness was a value-head failure, and the remaining gap to
   oracle is energy ties among near-duplicate edits, not budget
   (slack-insensitive at 0.45→0.475).

9. **Frozen-anchor outcome prediction ports to buffers** (edit_anchor):
   value-head planning 0.365 vs 0.115 base without value-gradient shaping
   (edit_valgrad 0.425 with it), and the anchor's chunk_pred loss trains
   to 0.005 — the predicted post-edit buffer embedding is nearly exact.

10. **The predictor is counterfactually grounded — and we can audit it
    directly** (`scripts/audit_counterfactual.py`: enumerate ALL feasible
    actions per state, predict each next-state latent, compare to
    symbolically executed ground truth). With LDAD or the frozen anchor,
    nearest-neighbor matching of F(s,a) to the true next states is
    0.99–1.00 (chance 0.29), RSA 0.80–0.93. Remove both grounding signals
    (disc_no_delta, base recipe) → matching 0.44; shuffle actions at
    training → 0.30 ≈ chance; value-only training → 0.25. One trace never
    shows two actions from the same state, so this grounding comes
    entirely from cross-problem generalization — see finding 13. A July 13
    fix changed only value-head audit conditioning; matching and RSA numbers
    remain valid, while old value-energy tau fields should be regenerated.

11. **Explicit counterfactual ranking is the strongest single lever
    found so far.** K alternative feasible actions per state, hinge-ranked
    by symbolic outcome against the executed one: disc_rank_k2 plans at
    **0.91 @strict / 0.985 @slack-2** (combo 0.635/0.89); K=2 already
    saturates (K=4: 0.925/0.98). Value-energy Kendall tau jumps 0.73 →
    0.94. Ranking *without* LDAD reaches 0.875 and matching 0.999 — explicit
    ranking and LDAD are two routes to the same grounding, and ranking is
    the stronger one. Combining ranking with the monotonicity hinge
    (disc_champion) gives 0.905/0.99 **and** the best raw-geometry energy
    (tau 0.82, oracle-goal planning 0.655 without any straightening).

12. **Ranking buys 1-step precision at the cost of depth calibration.**
    Lookahead-2 helps regression-trained values (mono_hi 0.655→0.735,
    valgrad 0.665→0.835) but *hurts* ranking-trained ones (rank_k2
    0.91→0.85): the margin loss perfects the ordering of 1-step candidates
    while distorting V's absolute scale, which multi-step cost sums rely
    on. Distillation targets and search depth interact.

13. **The historical scale sweep is consistent with cross-problem diversity
    helping, but is confounded.** Shrinking 100k indexed problems to 10k
    collapses combo to 0.135 @strict (matching
    0.82, tau 0.22); 30k → 0.265. Adding ranking at 10k doubles planning
    (0.26) and lifts tau to 0.53, but stays far from the 100k+rank 0.91.
    Removing within-trace distractors also costs (0.635→0.415). Because the
    three corpus sizes used roughly 1.6k/4.7k/15.6k optimizer updates, this
    does not isolate diversity from compute; a matched fresh-epoch rerun is
    required.

14. **Edit track completes the same picture at lower transition
    fidelity.** edit_rank is the new edit champion: 0.46/0.505 value
    planning (slack-sensitive at last), 0.60 with the goal-distance
    energy, tau 0.59. edit_distill_v matches the supervised value head
    without scalar defect-count targets (0.42 vs 0.425). Removing vandal
    negatives makes the value
    energy *anti-correlated* (tau −0.07) — negatives are not optional.
    The audit localizes the edit bottleneck: buffer matching is only
    0.39–0.47 (chance 0.10) vs 1.00 on discourse — one-step buffer
    prediction, not goal energy, is what limits the edit planner now.

15. **Depth-calibrated ranking (CostRanking) fixes the lookahead anomaly.**
    Ranking full MPC costs across depths — the executed 2-step
    continuation's cost (2 + V(F²)) against 1-step alternatives' (1 +
    V(F(alt))) — restores absolute calibration: rank_cal 0.905 look-1 →
    **0.945 look-2** (plain ranking degraded 0.91 → 0.85); 1000-episode:
    0.914/0.946. No new data needed (the 2-step continuation is already
    in every trace).

16. **Scalar progress labels are dispensable, but symbolic supervision is
    not yet eliminated.** Shape the metric
    with the monotonicity hinge (binary relevance labels), distill
    it into V, add binary-order ranking: disc_mono_distill_rank =
    **0.935 @strict / 0.99 @slack-2**, beating the scalar-label champion
    (0.905/0.99), with tau_V 0.953 and the second-best raw geometry
    (tau_G 0.813). Shaping is required: the same distillation on the
    unshaped metric gets 0.19. The full supervision ladder @slack-2:
    scalar+rank 0.99 = binary-tier 0.99 > scalar 0.935 > binary 0.885 >
    preference-label-free 0.775 > unshaped distill 0.64. Scalar
    remaining-steps labels are dispensable; binary relevance and symbolic
    counterfactual ordering are not. GAR is the active attempt to replace the
    latter with geometry from rendered counterfactual outcomes.

17. **Hierarchy pays at plan time, not train time.** Scoring 3-step
    sequences with one F_hi macro jump beats composing F three times for
    miscalibrated-value models (rank_k2: 0.970 @look-3-hier vs 0.915
    flat) — not by fidelity (F_hi matching 0.62–0.70 vs 0.77–0.91 for
    composed F) but by staying on the value head's training distribution.
    Once V is depth-calibrated the effect disappears (rank_cal: hier
    0.931 vs flat look-2 0.946 @1000 eps) — macro jumps and cost
    calibration are two routes to the same fix. Training-side, the
    hierarchy loss is neutral (combo_nohier 0.670/0.910 ≈ combo).

18. **The repository's hybrid displacement loss is not sufficient for
    stability.** It alone (no EMA, no stop-gradient, no VICReg) collapses:
    per-dimension state
    std 0.027, effective rank 80/256, value decodability 0.08; the
    stopgrad-only variant collapses too. And **VICReg is load-bearing**
    even with EMA+sg+anchor intact: removing it costs 0.635 → 0.27.
    This does not test Delta-JEPA's raw-action reconstruction faithfully and
    therefore is not a refutation of that paper's result.

19. **Neutral-to-positive architecture facts**: non-residual predictor
    0.710/0.920 (≥ combo — the residual skip is convenience, not
    mechanism, and LDAD is not trivialized by it: it constrains encoder
    displacements, not predictor outputs); hierarchy loss removable;
    straightening is seed-fragile (0.31–0.59 @strict across seeds) while
    hinge/ranking recipes are seed-stable.

20a. **The final recipe and its ablation matrix.** disc_mdr_cal (MDR +
    cost ranking) reaches **0.925 @strict / 1.000 @slack-2 / 0.945
    @look-2**. With ranking present the recipe is redundant: removing any
    single component (anchor, LDAD, VICReg, hinge, distillation,
    hierarchy, residual skip) costs only 2–7 points @strict and ~0
    @slack-2 — ranking + distilled-V is the irreducible core. Notably
    VICReg is catastrophic without ranking (0.27) but nearly free with it
    (0.895): the ranking gradients hold the space open. Only −LDAD
    degrades audit matching (0.94 vs 0.99+). MDR seed spread: 0.87–0.935.

20b. **Against parameter-matched outcome-candidate LM diagnostics.** The
    historical LM planners enumerate the same feasible actions as JEPA, but
    score rendered next-step candidates containing their computed values.
    They are privileged-candidate controls rather than exact interface
    parity. Token-level decoder-only LMs, same data, best LR each:
    2M → 0.315, 9M → 0.670, 27M → 0.735. Sentence-latent LMs (one latent
    per sentence, causal trunk, token decoder; 7.2M): 0.470 with decoder
    likelihood; adding a next-latent regression target lifts it to 0.510,
    and scoring by **latent distance beats the model's own decoder**
    (0.605) — the latent energy is the better ranker even in a
    reconstruction-based model. The naive JEPA (combo, 0.635) is on par
    with the matched token LM; the ranking-calibrated JEPA (0.935–0.948)
    beats the 3×-larger privileged diagnostic by ~20 points. That historical
    comparison does not survive the interface correction: the exact-interface
    9M token intent policy, which ranks the same outcome-free phrases as JEPA
    and appends the chosen intent plus observed outcome, reaches **0.827 ±
    0.003 strict / 0.978 ± 0.003 slack-2** over three seeds. This is the
    relevant token baseline and is much
    stronger than the 0.670 outcome-candidate result. The exact-interface
    sentence policy now reaches **0.690 ± 0.053 strict / 0.903 ± 0.039
    slack-2** over three seeds, substantially below the token policy. The
    sentence-plus-latent counterpart is also
    complete over three seeds. Intent-likelihood selection reaches
    **0.817 ± 0.034 strict / 0.953 ± 0.010 slack-2**; latent-distance
    selection is stronger at **0.840 ± 0.017 / 0.965 ± 0.013**, with exact
    strict values 0.830/0.860/0.830. Thus auxiliary next-latent prediction
    improves the sentence policy's likelihood selector by 0.127 strict on
    average, and latent-distance selection adds another 0.023. The token
    minimal JEPA is still completing its matched replication, so the final
    cross-architecture comparison remains open.

20c. **Cumulative-state targets are necessary** (disc_obs_only):
    replacing all state-prediction losses with anchored next-observation
    prediction destroys the dynamics (matching 0.22 ≈ chance, planning
    0.245). Together with the anchor's necessity for content, both
    prediction levels are needed — a hard constraint for any
    decoder-only redesign.

21. **Failures are chain-depth-limited.** Champion fail rate: 4.4%
    (3–4 necessary steps), 16% (5–6), 59% (7–9); apparent distractor
    effects are depth confounds. Depth is exactly what cost-calibrated
    look-2 buys (finding 15) and what the scaled-up domain will stress.

22. **The first combined non-symbolic reduction is selector-limited, not
    transition-limited.** Removing all seven seed-1-neutral objectives and
    the residual skip improves the geometry-distilled model from 0.750/0.940
    to **0.805 strict / 0.970 slack-2**, with counterfactual transition
    matching 0.997. Removing only its geometric preference objective collapses
    planning to 0.115/0.500 while matching remains 0.979 and value-order tau
    falls from 0.904 to 0.080. Against the matched symbolic-preference
    reference, transition matching is effectively tied (0.997 vs 1.000), but
    oracle-trace action top-1 is 0.948 vs 0.995 and strict success is 0.805 vs
    0.960. The mean trace has 4.07 necessary decisions and
    $0.948^{4.07}=0.804$, almost exactly the observed clean-model success.
    Closed-loop auditing sharpens the mechanism: useful-action top-1 is
    0.940 on clean histories but 0.706 after a prior distractor, versus
    0.989/0.857 under symbolic preferences. Errors are not predominantly
    ties (only 1.2% of competitive clean-model decisions have absolute margin
    below 0.02); symbolic labels widen the median useful-vs-distractor energy
    margin from 0.610 to 1.617. The remaining problem is therefore sharper,
    more off-trajectory-robust action ordering.

    The second removal round begins to assign mechanisms to the retained
    objectives. Removing one-step latent-state prediction reduces planning to
    **0.495/0.825**, transition matching to 0.265 (below the 0.287 random-match
    baseline), and value-order tau to 0.671: this is the transition-grounding
    objective. Removing on-trajectory outcome prediction instead gives
    0.755/0.945 with matching still 0.999 but tau 0.875; removing only its
    internal predicted-outcome rollout gives 0.770/0.935, matching 0.998, and
    tau 0.886. Those two auxiliaries organize the selection energy and improve
    off-trajectory robustness rather than identifying transitions. Their
    0.050/0.025 and 0.035/0.035 losses are in the predeclared second-seed band.
    The narrower predicted-outcome-rollout pair is now complete: its two-seed
    mean is 0.772/0.933 versus 0.797/0.968 for the reference, so its
    0.025/0.035 losses exceed the 0.020 removal threshold and the term is
    retained. The broader outcome-target pair is now complete too: its
    two-seed mean is 0.750/0.942, corresponding to losses of 0.048/0.025.
    Retain the full outcome target as well. Transition matching remains
    0.998--1.000 in both interventions, isolating their role to selection-
    energy organization rather than one-step transition identity.

## What the probes say the state actually is

Probe suite v2 (linear + MLP + random-encoder controls,
`runs/*/probe_v2.csv`; membership probes use [state; onehot(v)] binding
features):

- **A recency-weighted working memory, not a symbol table.** The current
  step's value is decodable at 0.90 (combo), then decays with lag:
  0.43 (1 step back) → 0.29 → 0.21 (chance 0.04). Older established
  values fade linearly but stay above chance (`figs/memory.pdf`).
- **Structure is objective-independent; content is objective-dependent.**
  Resolved-set membership (0.82), ancestor-of-query relevance (0.76),
  which-variable-just-resolved (0.33) are *identical across every
  training recipe* — base, no-LDAD, combo, geometry runs — while value
  decodability swings 0.39→0.90 with the anchor fix. The encoder learns
  the relational scaffold from text alone; the JEPA objectives decide
  what *content* survives in it.
- **No anticipatory answer computation.** Answer decodability from s_t is
  a step function: 0.06–0.18 mid-trace (tiny MLP>linear gap), snapping to
  1.00 exactly when the query variable resolves (`figs/emergence.pdf`).
  The base model can't even hold it then (0.33) — collusion, again.
- **Values are partially circularly coded**: ridge R² for cos(2πv/23)
  from s_t is 0.52 (combo) vs 0.20 (base) — the mod-p structure is
  partly geometric, and it tracks the anchor fix, echoing
  modular-arithmetic interpretability results.
- **Query identity is barely explicit** (0.19 vs 0.16 majority from s0)
  even though the relevance map is strong (0.76) — "what matters" is
  encoded relationally, not as a pointer.

## Stability

The main deterministic models with VICReg + EMA have per-dimension std near
1.0 and effective rank 220–243/256. Collapse does occur in explicit stability
ablations and in the new action-free latent-displacement pilot without an
appropriate target/regularizer combination; see the July 13 additions below.

## July 13 additions and current experiments

- **Fresh generation is implemented for new runs.** `FreshEpochSampler`
  changes each sample's global generator index by epoch and remains correct
  with persistent DataLoader workers. Historical results are unchanged and
  retain the fixed-corpus caveat above.
- **Data-only counterfactual control.** Each alternative action now carries
  its actual rendered outcome sentence, and a new objective predicts its
  frozen target embedding. K=1/2/4 runs test whether counterfactual data alone,
  without pairwise preference labels, explains the ranking gain.
- **Full probabilistic sentence-stream JEPA.** Prompt and solution sentences
  form one causal stream. The model learns an EMA diagonal-Gaussian target,
  posterior and prior distributions over an unobserved transition code, and a
  diagonal-Gaussian next-state predictor. It never reads intent phrases or a
  feasible-action interface. Stylized and official-iGSM runs are active.
- **Controlled probabilistic discourse JEPA separates action observability.**
  With inferred actions, the posterior transition code is strongly causal
  under shuffling (prediction error increases 7.24x) and linearly exposes the
  next outcome value at 0.992 accuracy. However, the state-conditioned prior
  exposes value at only 0.184 and retrieves the matching posterior with
  top-1 0.0015 / MRR 0.011. The posterior is therefore an outcome-residual
  code, not a usable pre-transition action proposal. In observed-action cells,
  pooled and token-concatenated intent encoders obtain calibrated Gaussian
  predictions (mean standardized residual squared 0.988--0.997; two-sigma
  coverage 0.955--0.969). Raw-action LDAD raises pooled displacement-value
  decodability 0.855 to 0.992 and state rank 88.1 to 107.1, but does not
  improve action sensitivity (1.395 to 1.305). With the order-preserving
  token-concatenated action encoder, LDAD instead improves matched prediction
  error 0.517 to 0.453, action sensitivity 1.151 to 1.269, action rank 4.60
  to 7.61, and displacement operation decoding 0.912 to 0.992. LDAD's gain
  therefore depends on action representation rather than universally
  improving probabilistic dynamics.
- **Latent-displacement factorial completed (30k problems, 10 epochs).** For
  EMA targets, no regularizer gives state/action-mean effective ranks 2.9/1.9;
  VICReg gives 125.1/9.2; SIGReg gives 128.8/12.8 (max 256/16). With a shared
  online target, the unregularized model fully collapses (state std 0.0009),
  VICReg remains low-rank (2.4/2.2), and SIGReg only partially rescues rank
  (16.8/7.7). Thus EMA + VICReg or SIGReg is needed for high-rank states in
  this formulation. Rank alone does not show that the inferred action is
  semantic.
- **Same-state counterfactual coverage separates posterior modes from a usable
  action prior.** With online stop-gradient + SIGReg on stylized iGSM, adding
  posterior-code displacement reconstruction improves matched probabilistic
  prediction L1 from 0.330 to 0.256 and posterior distinct-candidate matching
  from 0.285 to 0.645. Across 199 held-out states and every feasible rendered
  outcome, 64 prior samples cover 0.052/0.278 of geometrically distinct outcome
  regions without/with the auxiliary, and 0.155/0.494 of the posterior's
  learned modes. But only 0.004/0.016 of candidates receive a prior prediction
  as accurate as their correctly grounded outcome-informed posterior. The
  auxiliary therefore creates better separated posterior modes and some broad
  prior support, but not accurate pre-transition control. Prior-mean L1 remains
  0.489/0.488 and shuffled-prior sensitivity remains 1.001/1.001.
- **Stronger posterior--prior alignment collapses the action-free modes.**
  Removing the four-free-nats allowance while holding the action-KL weight at
  0.1 raises posterior--prior cosine to 0.996, but posterior distinct matching
  falls from 0.285 to 0.006 and prior distinct outcome coverage from 0.052 to
  0.010. Posterior-mode separation is only 9.2e-5; posterior and prior-mean
  prediction errors both become 0.518 with shuffled-action sensitivity 1.000.
  State std/rank remain 0.991/145.0, so their healthy appearance masks action-
  mode collapse. A stronger-KL follow-up is rejected by this controlled result.
- **A four-component action prior collapses to one component.** Holding the
  healthy online-stop-gradient+SIGReg model, action-KL, four free nats, and
  absence of posterior-code reconstruction fixed, a Gaussian-mixture prior
  achieves effective component use 1.36/4 and mean maximum component
  probability 0.924. Aggregate weights are [0.017, 0.021, 0.924, 0.038].
  Prior distinct coverage remains 0.050 versus 0.052 for one Gaussian and
  precise coverage 0.005 versus 0.004, while posterior distinct matching
  falls 0.285 to 0.230 and posterior-mode coverage 0.155 to 0.093. Extra prior
  capacity alone cannot identify unseen same-state modes; doing so likely
  requires counterfactual outcome-set training, a different supervision tier.
- **The official action-free target choice reverses what global rank suggests.**
  EMA+VICReg has healthy global state std/rank 0.988/151.2 and low demonstrated
  posterior/prior L1 0.0055/0.0063, but true outcomes of alternative actions at
  the same state are separated by only 0.005 L1: posterior distinct matching
  and prior distinct coverage are both zero. The online-stop-gradient+SIGReg
  control has lower global rank 40.9 but preserves same-state alternatives
  (separation 0.989, posterior distinct matching 0.753). Its prior samples cover
  0.598 of distinct outcome regions, yet only 0.073 at posterior-level accuracy;
  necessary actions receive 0.376 of assignments versus a 0.332 candidate
  share. Thus it learns broad mode support without an accurate or strongly
  goal-directed action prior. The exact posterior-code-reconstruction control
  is now complete: prior-mean/best-of-8 L1 improves from 0.966/0.601 to
  0.589/0.449 and broad distinct coverage from 0.598 to 0.675, but posterior
  L1 worsens from 0.006 to 0.049 and precise coverage falls from 0.073 to
  0.029. Reconstruction broadens support without producing accurate control.
  Full definitions and rows are in `runs/action_free_transfer.md`.
- **Faithful LDAD alone does not prevent probabilistic-state magnitude
  collapse.** In the observed-action variational JEPA with online
  stop-gradient and no VICReg/SIGReg, LDAD off/on gives state standard
  deviation 0.00075/0.00454 and action-sensitivity ratio 1.004/1.152. The
  LDAD-on model still reconstructs intents at 0.898 token and 0.582 exact-
  phrase accuracy, while its standardized residual squared is 1.012. A
  displacement decoder can therefore amplify small directional signals, and
  learned predictive variance can look calibrated when the target mean has
  collapsed. This is a substantive difference from the deterministic
  factorial, where faithful LDAD held the state space open without EMA.
  Fully online gradients reproduce the result: LDAD off/on gives state std
  0.00034/0.00305 and sensitivity 1.000/1.035, although the decoder reaches
  0.900 token and 0.583 exact-phrase recovery. Stop-gradient is therefore not
  the source of the variational failure. With EMA and no regularizer, LDAD
  changes state std/rank from 0.062/76.4 to 0.286/52.3 and action sensitivity
  from 1.182 to 1.455. This is better action dependence, but not a clean
  anti-collapse result because effective rank decreases. The complete 18-cell
  matrix identifies EMA+SIGReg+LDAD as the strongest balanced pooled-action
  setting: state std/rank 0.970/80.2, matched L1 0.062,
  shuffled/matched error 2.293, and standardized residual squared 0.983.
  EMA+VICReg+LDAD instead gives matched L1 0.018 and rank 90.1 but weaker
  action sensitivity 1.647.
- **Probabilistic architecture transfer separates fidelity from geometry.**
  At the selected EMA+SIGReg setting, mean-pooled residual prediction with
  raw LDAD gives matched L1 0.062, shuffled/matched error 2.293, and state
  rank 80.2. Replacing pooling with a two-dimensions-per-token ordered
  bottleneck is not better overall (0.102/2.086/rank 89.3). Removing the
  residual state skip changes the trade-off: the direct predictor with LDAD
  has the highest state rank and action sensitivity (171.8 and 2.990), but a
  larger matched L1 of 0.245. Relative to its exact LDAD-off control, raw
  LDAD improves direct-predictor error 0.501 to 0.245, sensitivity 1.728 to
  2.990, and rank 161.4 to 171.8. The EMA+VICReg pair confirms that this is
  not SIGReg-specific: direct prediction raises state rank from 97.2/90.1
  (residual, LDAD off/on) to 221.1/213.6, while LDAD improves the direct
  predictor's matched L1 0.469 to 0.320 and sensitivity 1.415 to 1.835.
  Thus ordered-token concatenation is not the mechanism; direct prediction
  preserves substantially richer state geometry, while the residual
  parameterization is easier to fit accurately. SIGReg yields stronger
  action dependence; VICReg yields the highest effective state rank. The
  official transfer strengthens the negative ordered-token conclusion: under
  EMA+SIGReg+LDAD, ordered versus pooled residual conditioning gives matched
  L1 0.081 versus 0.067, sensitivity 1.222 versus 1.462, and action-code rank
  7.8 versus 14.5. Ordered tokens improve exact phrase recovery only from
  0.730 to 0.775, so better auxiliary decoding does not imply a better
  transition control variable. The official VICReg direct pair is also
  complete: LDAD improves matched L1 0.612 to 0.199 and sensitivity 1.104 to
  1.307 while retaining state rank 197.3. Compared with official SIGReg
  direct LDAD (0.266/1.457/rank 163.5), VICReg favors fidelity and global
  rank while SIGReg favors action sensitivity.
- **Direct prediction does not rescue fully-online probabilistic training.**
  In the matched fully-online-gradient + SIGReg control, the residual
  predictor has state rank 14.1/17.5 with LDAD off/on. Replacing it by a
  direct predictor reduces rank further to 8.5/6.3; the observed-action code
  also collapses to effective rank 1.27/1.42. Its apparently large
  shuffled/matched error ratios (51.4/48.3) arise because matched L1 is only
  0.0037/0.0049 and almost all action dependence lies in one direction. Raw
  LDAD still recovers 0.869 token and 0.513 exact-phrase accuracy, but neither
  it nor the direct architecture replaces EMA stabilization.
- **Faithful probabilistic raw-action LDAD transfers to official iGSM.** In the
  exact pooled-action EMA+SIGReg pair, raw-action reconstruction changes
  matched prediction L1 from 0.463 to 0.067 and shuffled/matched action
  sensitivity from 1.105 to 1.462. State standard deviation remains healthy
  at 0.981/0.979, while effective rank decreases from 119.7 to 102.9. The
  LDAD-on decoder reaches 0.942 token accuracy and 0.730 exact-phrase recovery;
  standardized residual squared remains calibrated at 0.967 versus 0.991.
  This reproduces the stylized transition-grounding effect on independently
  rendered official text, but again shows that better action dependence need
  not increase global state rank.
- **The official effect does not depend on a residual predictor.** The exact
  official direct-predictor pair has healthy state std/rank 0.992/185.3
  without LDAD and 0.987/163.5 with it. Raw-action reconstruction lowers
  matched L1 from 0.719 to 0.266 and raises shuffled/matched sensitivity from
  1.097 to 1.457; token/exact-phrase recovery reaches 0.978/0.893 and
  standardized residual squared is 0.999. Direct prediction preserves richer
  global geometry than the residual pair but fits the conditional mean less
  accurately, reproducing the stylized fidelity--rank trade-off. The matched
  VICReg direct pair independently gives 0.612 -> 0.199 L1 and 1.104 -> 1.307
  sensitivity with LDAD, so the grounding effect is not SIGReg-specific.
- **Raw-action LDAD improves observed-action transition means; its accumulated
  uncertainty error is predominantly a horizon-dependent scale error.** With
  the true intent sequence fixed, 32 Gaussian trajectories are propagated
  recursively over 1,000 validation examples. On stylized iGSM,
  teacher-forced normalized L1 at H=1/H=4 changes from 0.396/0.431 without
  LDAD to 0.026/0.095 with it; on official iGSM it changes from 0.402/0.472
  to 0.051/0.071. Official LDAD-on open-loop L1 still grows
  0.052 -> 0.086 -> 0.167 -> 0.279 at H=1/2/4/8, while nominal two-sigma
  coverage falls 0.949 -> 0.921 -> 0.827 -> 0.779. A new held-out audit fits
  one standard-deviation multiplier per horizon on the first 500 trajectories
  and evaluates it on the disjoint remaining 500. At official H=8, the
  LDAD-on multiplier 1.684 changes residual squared 2.827 -> 0.997 and
  two-sigma coverage 0.777 -> 0.952. Conversely, the inaccurate no-LDAD model
  is over-dispersed and needs multiplier 0.612, changing coverage
  0.998 -> 0.953. Across all available horizons and both domains, calibrated
  residual squared is 0.969--1.079 and two-sigma coverage is 0.946--0.964.
  The fitted schedule is now frozen and evaluated on 1,000 disjoint test
  trajectories per model. At official H=8, test residual squared changes
  0.375 -> 1.003 without LDAD and 2.840 -> 1.002 with LDAD; corresponding
  two-sigma coverage changes 0.998 -> 0.953 and 0.776 -> 0.952. Across
  H=1/2/4/8, transferred test residual squared is 0.964--1.084 and coverage is
  0.943--0.962. Thus LDAD does not directly calibrate recursive spread, but
  its marginal coverage error is correctable by a validation-fitted scalar
  horizon schedule that transfers across splits; this does not remove
  accumulated mean bias. Full counts are in
  `runs/variational_rollout.md`.
- **Ordered-token action effects survive a severe bottleneck.** With the same
  16-dimensional final action code and the same probabilistic JEPA objective,
  shrinking each ordered token from 8 to 2 dimensions leaves the LDAD-on
  matched prediction error/shuffle ratio at 0.450/1.259, versus 0.453/1.269
  at width 8; token accuracy is 0.886 in both. Width 4 is also
  indistinguishable (0.456/1.257/0.887). The corresponding LDAD-off backbone
  sizes are 6.79M, 6.79M, and 6.81M parameters, versus 6.85M for mean
  pooling. This supports an ordered-token information effect rather than an
  extra-capacity explanation, but does not show that ordered tokens dominate
  pooling overall. The matched official transfer resolves that comparison in
  favor of pooling for dynamics: the width-2 ordered code has lower action
  rank and worse matched prediction and action sensitivity, both with and
  without LDAD, despite slightly better exact-phrase decoding.
- **The exact two-objective text Delta-JEPA control reproduces the core
  anti-collapse mechanism.** With fully online gradients, unnormalized latent
  MSE, and no EMA/stop-gradient/VICReg/SIGReg, latent prediction alone
  collapses to state std 0.00028 and counterfactual matching 0.307 (chance
  0.288). Reconstructing the observed intent phrase from the adjacent latent
  displacement raises state std/rank to 0.117/89.5 and matching/RSA to
  0.975/0.775; token/exact-phrase recovery is 0.909/0.584. Reconstructing four
  ordered observed phrases from the long displacement gives state std/rank
  0.080/140.1, matching/RSA 0.994/0.804, and token/exact-sequence recovery
  0.855/0.406. Thus the paper-style multi-action extension also prevents
  magnitude collapse and identifies counterfactual transitions; its stricter
  exact metric requires all four phrases to be correct.
- **First faithful observed-action LDAD result.** In the matched clean
  random-shooting H2/K2 model with hierarchy and rollout enabled, decoding
  the complete observed intent-token sequence from
  $s_{t+1}-s_t$ improves 0.670/0.895 to 0.720/0.940 and transition matching
  0.743 to 0.974. Value-order tau is unchanged (0.833 to 0.830), localizing
  the effect to action-conditioned transition grounding. This is a one-seed
  screen at the predeclared second-seed boundary, not yet a final result.
- **Official-iGSM faithful LDAD improves transition identity, not policy
  success, in the matched symbolic-preference pilot.** The LDAD run reaches
  0.535 strict / 0.805 slack-2, versus 0.410/0.735 for `real_rank` and
  0.460/0.765 for its 35M variant; transition matching is 0.956 and
  value-order tau is 0.796. The now-complete otherwise matched fresh-data
  no-LDAD run instead reaches 0.565/0.805, matching 0.889, tau 0.791, and RSA
  0.683 versus 0.786 with LDAD. Thus raw-action reconstruction strongly grounds
  the transition without improving selection; it is removed from this pilot
  configuration under the fixed shear rule. The matched closed-loop audit
  agrees: clean/after-error useful-action top-1 is 0.848/0.653 without LDAD
  versus 0.837/0.623 with it, and failure in the highest distractor-load
  stratum falls from 0.789 to 0.632 when LDAD is removed.
- **The official pilot is distractor-load and recovery limited.** Across 200
  held-out validation episodes, strict success is 0.535. Competitive
  useful-action top-1 is 0.837 on clean histories but 0.623 after a previous
  distractor. Failure rises from 0.225 with 0--4 distractor variables to
  0.589 with 5--9 and 0.789 with 10--20; necessary-chain buckets are much
  flatter at 0.412/0.489/0.500 for 1--4/5--8/9--15 steps. Only 2.3% of
  competitive decisions are within an absolute energy margin of 0.02. Thus
  this symbolic-preference checkpoint is limited by sharp ordering under
  distractor load and off-history recovery, not predominantly by near ties or
  transition length. The matched no-LDAD audit improves clean/after-error
  top-1 to 0.848/0.653 and reduces high-distractor failure to 0.632, so LDAD
  does not explain or repair this selector limitation.
- **Transition identity and planning selection separate in the greedy H2
  screen.** The final all-objective checkpoint reaches 0.750 strict / 0.940
  slack-2, whereas the seed-1 no-LDAD removal reaches 0.755/0.960. At matched
  seed 2, however, removing LDAD lowers planning from 0.785/0.960 to
  0.720/0.920 and transition matching from 0.978 to 0.799. The two-seed
  LDAD/no-LDAD means are 0.768/0.738 strict and 0.950/0.940 slack-2, a mean
  LDAD contribution of +0.030/+0.010 with a sign reversal across seeds.
  Therefore LDAD is not safely removable from the over-complete model.
  Across seed-1 LDAD weights, 0.05
  gives 0.775/0.940 with 0.936 matching, while weight 1.0 gives 0.755/0.940
  with 0.997 matching. Stronger action reconstruction monotonically improves
  transition identification in these endpoints but not episode success. An
  exact two-seed LDAD add-back to the reduced model reaches 0.770/0.955 and
  0.805/0.960, a mean 0.788/0.958 versus 0.798/0.968 without it. Removing the
  auxiliary improves both budgets by 0.010, so faithful LDAD is excluded from
  the selected planner despite its representation-level benefit.
- **Geometry-to-value distillation is seed-sensitive and requires a reduced-
  model resurrection test.** At seed 1, removing scalar regression from goal
  distance to the learned energy improves strict planning 0.750 to 0.800 but
  lowers slack-2 0.940 to 0.920. At matched seed 2, the same removal instead
  drops strict planning 0.785 to 0.680 and slack-2 0.960 to 0.945. Averaging
  the paired seeds gives reference/removal 0.768/0.740 strict and 0.950/0.933
  slack-2: losses 0.0275/0.0175, still inside the predeclared ambiguity band.
  The objective must not be called removable from this evidence. An exact
  add-back to the reduced model is active to test the relevant interaction.
- **A grounded transition model still needs a learned selector.** In the
  reduced second-round model, removing latent-goal preference distillation
  leaves counterfactual transition matching at 0.979 but reduces value-order
  tau to 0.080 and planning to 0.115 strict / 0.500 slack-2. The exact reduced
  reference is still training, so its numerical ablation loss is pending, but
  this preference-free endpoint is already close to random action selection
  (0.055/0.405) despite identifying action-conditioned consequences. The
  geometric preference objective is therefore retained as the component that
  converts grounded dynamics into goal-directed decisions.
- **Lowering the geometric tie filter adds harmful preferences.** In the
  reduced model, changing only the latent-distance label gap from 0.02 to
  0.005 reduces strict/slack-2 success from 0.805/0.970 to 0.740/0.925.
  Transition matching is unchanged at 0.997, but value-order tau falls
  0.904 -> 0.874, clean-history useful-action top-1 falls 0.940 -> 0.914,
  and median useful-vs-distractor margin falls 0.610 -> 0.520. The additional
  small-distance pairs are noisy rather than missing useful supervision; the
  0.02 geometric tie filter is retained.
- **Reduced-model geometry-to-value regression does not repair the selector.**
  At seed 1, adding it back changes strict/slack-2 from 0.805/0.970 to
  0.760/0.950. At matched seed 2 it changes 0.790/0.965 to 0.740/0.970.
  Two-seed means are therefore 0.798/0.968 without scalar regression and
  0.750/0.960 with it, a decisive 0.048 strict loss. Mean transition matching
  remains 0.999 while value-order tau is only 0.844. The term remains removed.
  At seed 2, clean/after-error top-1 changes from 0.933/0.758 without the
  scalar target to 0.913/0.781 with it: improved recovery does not compensate
  for worse clean decisions.
- **Doubling off-trajectory training exposure does not improve recovery.**
  Raising the distractor-example probability from 0.15 to 0.30 changes
  strict/slack-2 success from 0.805/0.970 to 0.790/0.935, transition matching
  from 0.997 to 0.995, and value tau from 0.903 to 0.882. Clean/after-error
  useful-action top-1 also falls from 0.940/0.706 to 0.933/0.649. The default
  0.15 mixture is retained.
- **Preference-margin tuning does not improve action selection.** At seed 1,
  the default margin 0.5 scores 0.805/0.970 and a softer 0.25 margin scores
  0.800/0.985. The soft-margin result reverses at matched seed 2:
  0.790/0.965 for margin 0.5 versus 0.725/0.950 for margin 0.25. Two-seed
  means are 0.798/0.968 and 0.763/0.968, respectively, so the 0.035 strict
  loss rejects margin 0.25. At seed 2, the softer margin changes
  clean/after-error top-1 from 0.933/0.758 to 0.900/0.776: it slightly
  improves recovery while damaging the more frequent clean decisions.
  Larger margins 1.0 and 2.0 degrade seed-1
  planning to 0.575/0.870 and 0.435/0.875.
  Their median useful-action energy gaps nevertheless grow to 0.920/1.477
  while clean-history top-1 falls to 0.844/0.749: the larger hinge produces
  confident misordering, not better separation.
  Margin 0.5 is retained; every tested alternative is rejected.
- **Combining the first accepted removals improves the non-symbolic planner.**
  The seed-1 reduced model reaches 0.805 strict / 0.970 slack-2, compared with
  0.750/0.940 for the deliberately over-complete H2 reference. It uses direct
  next-state prediction, one-step latent prediction, on-trajectory outcome
  prediction, H2 geometric preference distillation, VICReg, and EMA; it omits
  symbolic ranking, faithful LDAD, terminal monotonicity, scalar geometry-to-
  value regression, counterfactual transition prediction, hierarchy, the
  explicit open-loop latent loss, and the residual state skip. This validates
  Across three seeds the reduced model reaches 0.797 ± 0.008 / 0.963 ± 0.008.
  Joint shearing therefore helps but does not yet win the matched comparison:
  the token LM is
  0.827 ± 0.003 / 0.978 ± 0.003 and the sentence-plus-latent baseline reaches
  0.840 ± 0.017 / 0.965 ± 0.013 using latent-distance selection. Transition
  matching is 0.997, value-order tau is
  0.904, and local action top-1 is 0.948. The compounding estimate
  $0.948^{4.07}=0.804$ nearly equals observed strict success 0.805, localizing
  the remaining failure to action ordering rather than transition identity.
  The exact seed-2 reference reaches 0.790/0.965 with transition matching
  0.999 and clean/off-history useful-action top-1 0.934/0.758, giving a
  two-seed reference mean of 0.798/0.968. The reduced geometry-to-value and
  faithful-LDAD add-backs are both rejected by matched replications; only the
  bounded-beam selector gate remains active.
- **Faithful LDAD does not replace all stabilization in the full model.** In
  the deliberately extreme online-gradient/no-VICReg control, LDAD improves
  transition matching 0.456 to 0.701 and state effective rank 79.1 to 107.5,
  yet planning falls from 0.565/0.795 to 0.440/0.785 because value-order tau
  falls from 0.793 to 0.705 (top-1 0.870 to 0.799). This separates transition
  identity from energy calibration. The cleaner reduced-model target
  replacement independently retains EMA at 0.720/0.910 versus 0.805/0.970;
  the extreme joint removal is not itself a one-component shear.
- **Observed-action LDAD factorial, EMA and stop-gradient rows.** In the isolated
  one-step JEPA with no variance regularizer, raw-token LDAD raises transition
  matching from 0.432 to 0.959 and RSA from 0.300 to 0.737; the decoder reaches
  0.910 token accuracy and 0.584 exact-phrase recovery. EMA+VICReg already
  obtains 0.998 matching without LDAD (0.992 with it), and EMA+SIGReg obtains
  0.990 without it. Thus LDAD supplies transition grounding when the weak cell
  lacks it but is redundant for that metric once a strong regularizer has
  organized the representation. With an online stop-gradient target and no
  regularizer, the no-LDAD model nearly collapses (state std/rank
  0.102/2.95, matching 0.299); faithful LDAD raises these to
  0.343/101.8 and 0.963. Thus observed-action decoding can replace EMA and
  the explicit regularizer in this controlled stop-gradient setting, although
  state scale remains below the healthier unit-scale VICReg/SIGReg cells.
  Even with fully online gradients, no stop-gradient, and no regularizer,
  LDAD changes state std/matching from approximately 0.000/0.322 to
  0.145/0.988 (state rank 90.3). Thus faithful LDAD can prevent complete
  collapse and identify transitions on its own, but does not enforce the
  unit-scale geometry or planning-energy calibration supplied by other terms.
  The final fully-online VICReg pair makes the separation especially clear:
  VICReg without LDAD has state std/rank 1.016/234.5 but only 0.334 matching;
  adding LDAD gives 1.007/219.6 and 0.980 matching (RSA 0.558 to 0.909).
  Distributional health is therefore not sufficient for action-conditioned
  transition grounding.
- **GAR-teacher audit.** On the exact 100 held-out H2/K2 training-style
  candidate sets, the random-shooting geometric labels order 95.9% of the
  oracle-distinct pairs correctly when decisive, but cover only 73.3% of
  those pairs; 20.9--21.4% of emitted preferences separate oracle-tied
  actions. Raw-text LDAD does not change the teacher construction or consume
  oracle labels, but it can change target geometry and therefore the emitted
  preferences. The existing off/on teacher audits used unequal sample counts,
  so they are not a valid paired estimate of that indirect effect. LDAD's
  direct controlled benefit is transition identity; teacher quality and
  student fitting must be reported separately. `scripts/eval_run.sh` now runs
  this diagnostic automatically with a consistent configuration for every
  new GAR checkpoint.
- **The first geometry-greedy horizon result does not reward a longer
  continuation.** H=4/K=2 reaches 0.695 strict / 0.930 slack-2, with
  transition matching 0.883 and value-order tau 0.792. Its teacher is fairly
  accurate overall (oracle top-1 0.900), but the trace-with-distractors stratum
  falls to top-1 0.722 and pairwise tau-a 0.228, versus 1.000/0.898 on clean
  traces. Longer teacher continuation therefore does not by itself cure the
  trace-terminal goal problem, and its labels are not yet translating into a
  better student energy.
- **A continuation is necessary for the non-symbolic geometric teacher.**
  H=1/K=2, which scores the immediate child without any continuation, reaches
  only 0.315 strict / 0.800 slack-2, transition matching 0.887, and value-order
  tau 0.542. Its teacher covers 55.2% of oracle-distinct pairs and is 0.879
  accurate when decisive, but selects an oracle-best root action only 0.750 of
  the time; the learned value top-1 is 0.726. This isolates the teacher
  horizon from the independent hierarchy and rollout objectives and shows why
  H=2 is the relevant minimum rather than an arbitrary “depth option.”
- **Planner depth greater than one is an oracle-action diagnostic, not a fair
  latent-search result.** The current planner uses the reference dependency
  graph to enumerate future feasible actions and detect terminal sequences.
  On the reduced model this produces the visibly non-monotonic lookahead-1/2/4
  strict curve 0.805/0.495/0.910; the depth-4 gain partly exposes symbolic
  solution structure rather than model-only imagination. Lookahead 1 already
  enumerates every currently feasible intent and remains the main protocol.
  Deeper runs now require `allow_oracle_future_actions=true`, carry an
  `_oracle_actions` filename suffix, and are excluded from fair comparisons.
- **H=2 remains optimal for a width-one greedy teacher when the checkpoint is
  held fixed.** Recomputing only
  the current reduced checkpoint's geometry-greedy labels at H=1/2/4/8/16
  gives oracle top-1 0.70/0.86/0.81/0.80/0.80 and pairwise tau-a
  0.354/0.720/0.677/0.616/0.604. Pair coverage is
  0.768/0.841/0.884/0.848/0.884, so longer horizons do not fail merely by
  filtering too many comparisons. This matched audit separates intrinsic
  teacher quality from student retraining and confirms H=2 as the best
  width-one continuation horizon.
- **The geometric continuation curve peaks at H=2.** H=8/K=2 falls to 0.620
  strict / 0.885 slack-2, transition matching 0.901, and value-order tau
  0.728. Although teacher pair coverage rises to 0.921, decisive pair accuracy
  falls to 0.836 and oracle top-1 to 0.830. Together with H=1 at 0.315/0.800
  and H=4 at 0.695/0.930, this shows an interior optimum: one continuation
  step resolves immediate geometric ambiguity, while longer greedy
  continuations accumulate errors relative to the trajectory-specific goal.
- **A bounded geometric beam recovers much of the longer-horizon teacher
  loss.** On the same reduced checkpoint and the same 200 anchors, H=4 with
  beam width B=1/2/4/8 gives oracle top-1 0.810/0.830/0.870/0.900 and
  pairwise tau-a 0.646/0.641/0.701/0.728. H=8 improves from
  0.805/0.580 at B=1 to 0.900/0.670 at B=8; H=16/B=8 reaches
  0.895/0.672. The paired H=2/B=1 reference is 0.865/0.661. Thus an early
  greedy continuation error, not horizon alone,
  explains part of the old degradation. Beam expansion still uses only
  feasible actions, rendered outcomes, and EMA geometry; symbolic remaining
  steps are audit-only. Reduced-model H=4/B=8 and H=8/B=8 end-to-end cells
  are active. H=16/B=8 does not improve the teacher enough to justify its
  greater cost and remains an audit-only horizon.
- **Increasing the number of geometric root alternatives is also harmful.**
  At H=2, K=2/4/8 gives strict success 0.750/0.740/0.710 and slack-2
  0.940/0.935/0.910. Teacher top-1 declines 0.930/0.920/0.890, pair coverage
  0.836/0.743/0.696, and teacher tau-a 0.776/0.694/0.615. K=8 still has
  counterfactual transition matching 0.988, so the degradation is in the
  growing set of noisy geometric preferences rather than transition
  identity. K=2 is retained.
- **On-trajectory outcome prediction is retained for selection rather than
  transition identity.** Removing it gives 0.680 strict / 0.940 slack-2, a
  0.070 strict loss relative to the final H2 reference, while transition
  matching actually rises to 0.998. Value-order tau falls from 0.847 to 0.800.
  This is further evidence that counterfactual transition identifiability and
  calibrated action selection are distinct requirements.
- **VICReg remains necessary in the selected planning model.** Its exact
  removal falls from 0.750/0.940 to 0.455/0.765, transition matching falls
  from 0.986 to 0.765, and value-order tau falls from 0.847 to 0.680. Because
  faithful LDAD remains active in this ablation, the result directly rules out
  the claim that action reconstruction alone replaces distributional
  regularization and energy organization in the full planning objective.
- **The reduced model independently confirms VICReg is load-bearing.** Its
  exact removal changes strict/slack-2 planning from 0.805/0.970 to
  0.615/0.755, transition matching from 0.997 to 0.953, and value-order tau
  from 0.904 to 0.790. Clean-history useful-action top-1 remains 0.870 but
  falls to 0.336 after an earlier distractor. Thus variance--covariance
  regularization organizes both counterfactual transitions and off-history
  selector recovery; its role is not exhausted by maintaining global scale.
- **EMA remains load-bearing after joint shearing.** In the reduced model,
  replacing the EMA target by an online stop-gradient target changes
  strict/slack-2 planning from 0.805/0.970 to 0.720/0.910, transition matching
  from 0.997 to 0.987, and value tau from 0.904 to 0.871. Clean/off-history
  useful-action top-1 falls from 0.940/0.706 to 0.908/0.562. Both budget losses
  exceed the one-seed retention boundary, so EMA is retained without another
  replication.
- **Direct next-state prediction outperforms the residual transition
  parameterization in the over-complete objective set.** Removing the skip in
  $\hat{s}_{t+1}=s_t+F(s_t,u_t)$ improves planning from 0.750/0.940 to
  0.800/0.960, transition matching from 0.986 to 0.995, and value-order tau
  from 0.847 to 0.891. This is architecture-by-objective evidence, not a
  universal claim: the final selected objective set requires its own matched
  residual/direct ablation.
- **EMA remains necessary in the first clean selection round.** Replacing the
  EMA target with an online stop-gradient target while holding the other
  objectives fixed reduces strict/slack-2 planning from 0.750/0.940 to
  0.575/0.840, transition matching from 0.986 to 0.943, and value-order tau
  from 0.847 to 0.735. The combined round-one model therefore retains EMA but
  removes LDAD, terminal monotonicity, geometry-to-value regression,
  counterfactual transition prediction, hierarchy, open-loop latent
  prediction, and the residual transition skip. It is active as
  `disc_latent_goal_h2_r1`. Its second-round exact removals retain latent
  prediction, outcome prediction, VICReg, geometric preference learning, EMA,
  and the predicted-outcome rollout internal to the outcome loss. The two
  ambiguous outcome interventions were resolved by matched two-seed means:
  removing the full target loses 0.048/0.025 strict/slack-2, while removing
  only its internal rollout loses 0.025/0.035.
- **Trace-terminal monotonicity is removed.** Retaining the constraint with
  zero margin reaches 0.765 strict / 0.940 slack-2; removing it entirely
  reaches 0.780/0.950. The combined model and every second-round ablation set
  its weight to zero.
- **First geometry-greedy H2 removal result.** Removing auxiliary
  counterfactual transition prediction (while retaining the independent GAR
  candidates) reaches 0.755 strict / 0.935 slack-2, transition matching
  0.988, and value-order tau 0.858. Local counterfactual value top-1 is 0.927;
  compounding this across the mean 4.07 necessary choices predicts
  $0.927^{4.07}=0.735$ strict success, close to the observed 0.755. The
  component satisfies the predeclared removal rule and is absent from the
  combined model.
- **GAR goal-definition audit (500 anchors).** The original full
  trace-terminal goal reaches teacher/oracle tau-a 0.591 and top-1 0.842.
  An audit-only necessary-trace terminal, constructed with symbolic ancestry,
  improves this to 0.623/0.870 and removes most of the degradation specific to
  distractor-containing demonstrations. This identifies terminal distractor
  content as one source of geometric-label noise, but is not a permissible
  non-symbolic training recipe. A deployable prompt-plus-final-sentence goal
  is strongly anti-informative (tau-a -0.084, top-1 0.454), so no GPU run was
  launched for it. The matched geometry-greedy horizon comparison is complete
  for H={1,2,4,8,16}. H=1 is the no-continuation control. H=16 reaches
  0.655 strict / 0.895 slack-2, teacher top-1 0.790, transition matching
  0.912, and value-order tau 0.768. Clean/after-error useful-action top-1 is
  only 0.870/0.568. It remains substantially worse than H=2 and confirms
  accumulated width-one teacher error.
  A second trace-only control uses four independently completed trajectories
  from a uniformly random feasible policy. Averaging their goal latents gives
  tau-a 0.285/top-1 0.64 and nearest-set distance gives 0.339/0.65, versus
  0.679/0.89 for the demonstrated goal on the same 100 anchors. Thus terminal
  latents are trajectory-specific rather than interchangeable; this negative
  audit was not promoted to a training experiment.
- **Audit correction.** Value-energy counterfactual scores must condition the
  value head on the initial prompt/buffer state. The script now does so.
  Transition matching, RSA, and goal-distance scores were unaffected. The
  corrected `edit_attn` audit is match 0.497, RSA 0.899, value tau 0.612;
  older value-energy audit fields should be regenerated before citation.
- **Matched official-iGSM autoregressive baselines are complete at one
  validation seed.** Random feasible selection reaches 0.200/0.460
  strict/slack-2; token intent likelihood 0.415/0.675; sentence intent
  likelihood 0.450/0.755; and sentence likelihood with auxiliary next-latent
  prediction 0.455/0.765. Using that auxiliary through latent-distance
  selection instead reaches 0.445/0.715. Thus sentence likelihood is the
  strongest label-free official baseline so far, while the latent target adds
  only 0.005/0.010 and is not itself a better selection metric. The current
  non-symbolic official transfer remains active; the symbolic-preference
  reference is 0.565/0.805.

- **The stylized symbolic-preference ceiling is stable over three seeds.**
  Exact-interface strict/slack-2 success is 0.960/0.995,
  0.950/0.995, and 0.975/1.000, giving **0.962 ± 0.013 / 0.997 ± 0.003**.
  The remaining gap to the non-symbolic model therefore cannot be attributed
  to an unusually favorable annotated-reference seed; it is a persistent
  action-ordering gap.

## July 14 hierarchy update

- **Dense recursive supervision improves dynamics at both levels.** Applying
  the shifted-sequence loss at every origin lowers high-level three-macro-step
  LN-L1 error from .684 to .622 at supervised depth 3. At the primitive level,
  depth 4 lowers eight-step error from .867 to .614 and improves flat strict
  success on the same 200 long-tail problems from .300 to .355. Depth 8 has
  slightly better eight-step error (.604) but lower success (.345), so
  prediction error alone is not a sufficient selector.
- **Macro-state value is better calibrated than terminal latent distance.** On
  11,659 valid macro spans, encoded terminal-state distance has correlation
  .478, pair accuracy .591, top-1 .590, and regret .545. Exact remaining-cost
  regression on the predicted next state reaches .860/.734/.800/.280.
  Adding a pairwise ranking hinge changes pair accuracy to .735 but leaves
  top-1 and regret unchanged. These value rows use oracle graph supervision
  and are mechanism diagnostics, not non-symbolic headline results.
- **Continuous latent-action hierarchy does not confirm.** On the
  100-problem selection screen, learned state-value planning gives strict
  success .26/.30/.33 at one/two/three macro steps, while HWM-style encoded
  terminal distance stays .31/.31/.31 and flat control is .28. Three paired
  500-problem confirmations reverse the apparent gain: flat
  .314/.336/.322 versus hierarchy .272/.320/.274, for means .324 versus .289.
  Continuous macro-vector CEM followed by low-level subgoal refinement is
  rejected; this result does not apply to discrete macro-action sequences.
- **CEM under-budgeting and simple off-support search are ruled out as primary
  explanations.** Reported planning uses 1,200 candidates, 20 refits, 10
  elites, and variance EMA .9. Action banks, hard projection, prior-noise CEM,
  local GMMs, learned support, measured open-loop reachability, and ensemble
  disagreement do not yield a confirmed gain.
- **Exact controller outcomes reveal headroom, but learned rerankers do not
  recover it.** Reranking the top 32 continuous candidates by exact remaining
  work after three actual lower-controller steps raises strict success from
  .26 to .44 and lowers distractor selection from .754 to .192. Endpoint
  reachability remains .26 and latent-goal reranking reaches at most .30.
  Learned regression/pairwise/listwise outcome heads score .26/.26/.29 versus
  .31 flat on a held-out split despite regression correlation .794. The
  failure is task-progress scoring, not absence of useful candidates.
- **The apparent support-constrained discrete gain is an ordering artifact.** The planner encodes
  actual intent-phrase sequences as macro actions, proposes future elements
  from unresolved problem actions without oracle future-feasibility labels,
  directly executes the first currently feasible action, and replans. The
  first implementation lexicographically capped its exponentially expanding
  sequence list, over-representing early first actions. Its H2 mean .376
  versus .326 flat and H3 mean .386 versus .311 are invalidated despite
  replication. With equal proposal multiplicity per first action, arbitrary
  H2 sequence planning scores .110 versus .310 flat on the selection split;
  a learned-support beam scores .150 and reaches .250 with a unit-weight
  one-step value safeguard. The always-first control is .100. No positive
  discrete hierarchy result is currently supported.
- **Faithful iGSM also leaked action order.** The official graph iterates
  parameters in a near-topological order; before correction, always taking the
  first feasible action solves 100/100 long problems. `FaithfulProblem` now
  exposes a stable problem-specific shuffled action menu to every model.
  Historical faithful planning results using reference order must be rerun.
- **Corrected support-constrained hierarchy is a replicated negative.** A
  planning-matched support head and prefix-aware macro value reach .320--.330
  on the 100-problem selection screen only with a strong flat one-step value
  safeguard. Frozen evaluation on three fresh paired 500-problem sets gives
  hierarchy .310/.326/.296 versus flat .318/.332/.300: means .311 versus .317
  and deltas -.008/-.006/-.004. High-level dense supervision improves
  recursive prediction, but this controller adds no execution benefit.

## Open directions

- **Close the easy-domain paper matrix, then change hierarchy domain**: the
  corrected easy-domain 3×500 hierarchy gate failed. Complete the flat
  three-seed cumulative ladder, matched ablations, curves, and probes; study
  temporal hierarchy next where token/phrase/sentence boundaries provide a
  meaningful abstraction target.
- **Retain the continuous outcome result as a diagnostic**: the exact
  realized-progress upper bound is informative, but learned controller-outcome
  reranking is not the selected mechanism and should not receive more tuning
  before faithful discrete transfer.
- **Probabilistic follow-up**: the validation-fitted spread schedule transfers
  to frozen test sets, and the official posterior-code reconstruction control
  is complete. Seek a goal-conditioned or counterfactual-set-trained
  action-free prior rather than optimizing only broad same-state coverage.
- **Controlled data scaling**: matched optimizer steps with the new
  fresh-per-epoch sampler.
- **Edit transition fidelity**: buffer matching is 0.4 vs 1.0 on
  discourse (finding 14) — a slot-aligned predictor or per-sentence
  outcome anchors for the *changed* slot are the obvious attack.
- **Close the non-symbolic selector gap**: the reduced model averages
  0.797 strict versus 0.827 for the three-seed token intent policy and 0.962 for the
  three-seed symbolic-preference reference. Transition matching is already 0.997; the
  remaining target is useful-action ordering and recovery after a detour.
- Harder worlds: deeper DAGs, larger moduli, compositional templates,
  natural-language surface variation; scale d_model/params.
- Variational action prior p(a | s, g) for sampling actions without
  enumerating the interface (open-ended generation).
