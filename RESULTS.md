# Results — five experimental cycles (2026-07-09/10)

Setup: iGSM-style synthetic reasoning (mod-23 arithmetic DAGs, 6–12
quantities, 3–9 necessary steps, distractor variables). All models ~9M
params, 20 epochs on fresh data (V100/RTX6000, ~40–70 min per run).
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

## The twenty findings

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

4. **Delta-JEPA's displacement loss is load-bearing** (replicates
   arXiv:2606.31232 in language): removing it halves planning (35→15%)
   and pushes chosen-action distractor rates to near-random, while op
   identity stays 100% decodable from Δs when present.

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
    entirely from cross-problem generalization — see finding 13.

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

13. **Cross-problem diversity is the counterfactual fuel; ranking is a
    sharpener, not a substitute.** Shrinking 100k unique problems to 10k
    (same optimizer steps) collapses combo to 0.135 @strict (matching
    0.82, tau 0.22); 30k → 0.265. Adding ranking at 10k doubles planning
    (0.26) and lifts tau to 0.53, but stays far from the 100k+rank 0.91.
    Removing within-trace distractors also costs (0.635→0.415).

14. **Edit track completes the same picture at lower transition
    fidelity.** edit_rank is the new edit champion: 0.46/0.505 value
    planning (slack-sensitive at last), 0.60 with the goal-distance
    energy, tau 0.59. edit_distill_v matches the supervised value head
    label-free (0.42 vs 0.425). Removing vandal negatives makes the value
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

16. **The non-symbolic value head wins (headline).** Shape the metric
    with the monotonicity hinge (binary relevance labels only), distill
    it into V, add binary-order ranking: disc_mono_distill_rank =
    **0.935 @strict / 0.99 @slack-2**, beating the scalar-label champion
    (0.905/0.99), with tau_V 0.953 and the second-best raw geometry
    (tau_G 0.813). Shaping is required: the same distillation on the
    unshaped metric gets 0.19. The full supervision ladder @slack-2:
    scalar+rank 0.99 = binary-tier 0.99 > scalar 0.935 > binary 0.885 >
    label-free 0.775 > unshaped distill 0.64. Scalar remaining-steps
    labels are dispensable.

17. **Hierarchy pays at plan time, not train time.** Scoring 3-step
    sequences with one F_hi macro jump beats composing F three times for
    miscalibrated-value models (rank_k2: 0.970 @look-3-hier vs 0.915
    flat) — not by fidelity (F_hi matching 0.62–0.70 vs 0.77–0.91 for
    composed F) but by staying on the value head's training distribution.
    Once V is depth-calibrated the effect disappears (rank_cal: hier
    0.931 vs flat look-2 0.946 @1000 eps) — macro jumps and cost
    calibration are two routes to the same fix. Training-side, the
    hierarchy loss is neutral (combo_nohier 0.670/0.910 ≈ combo).

18. **The Delta-JEPA stability claim does not transfer to language.**
    LDAD alone (no EMA, no stopgrad, no VICReg) collapses: per-dim state
    std 0.027, effective rank 80/256, value decodability 0.08; the
    stopgrad-only variant collapses too. And **VICReg is load-bearing**
    even with EMA+sg+anchor intact: removing it costs 0.635 → 0.27.
    Stability in this domain needs the full kit.

19. **Neutral-to-positive architecture facts**: non-residual predictor
    0.710/0.920 (≥ combo — the residual skip is convenience, not
    mechanism, and LDAD is not trivialized by it: it constrains encoder
    displacements, not predictor outputs); hierarchy loss removable;
    straightening is seed-fragile (0.31–0.59 @strict across seeds) while
    hinge/ranking recipes are seed-stable.

20. **Failures are chain-depth-limited.** Champion fail rate: 4.4%
    (3–4 necessary steps), 16% (5–6), 59% (7–9); apparent distractor
    effects are depth confounds. Depth is exactly what cost-calibrated
    look-2 buys (finding 15) and what the scaled-up domain will stress.

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

No collapse in any run: VICReg + EMA targets held per-dim std ≈ 1.0 and
effective rank 220–243/256 throughout; FSQ/discrete actions not yet needed.

## Open directions

- **Depth-calibrated ranking**: rank multi-step rollout costs (not only
  1-step candidates) so lookahead helps ranking models too (finding 12);
  target ≥0.95 @strict with look-2.
- **Edit transition fidelity**: buffer matching is 0.4 vs 1.0 on
  discourse (finding 14) — a slot-aligned predictor or per-sentence
  outcome anchors for the *changed* slot are the obvious attack.
- **Close the last gap to oracle** (0.905→1.0): error analysis of the
  ~10% failures (long chains? mul-heavy? tie patterns).
- Use the trained hierarchy (macro-actions + F_hi) in the planner: propose
  K-step macro rollouts, refine with the low-level model (HWM-style).
- Harder worlds: deeper DAGs, larger moduli, compositional templates,
  natural-language surface variation; scale d_model/params.
- Variational action prior p(a | s, g) for sampling actions without
  enumerating the interface (open-ended generation).
