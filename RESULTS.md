# Results — first two experimental cycles (2026-07-09)

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
| disc_valgrad | 0.665 | **0.835** | 0.920 | **0.07** (look-2) |
| **disc_mono_hi** | **0.655** | — | **0.935** | 0.15 |

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

*disc_mono_novalue's value head is untrained; its 0.535 is the value
column's floor, not a real planner.

## The nine findings

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

## Stability

No collapse in any run: VICReg + EMA targets held per-dim std ≈ 1.0 and
effective rank 220–243/256 throughout; FSQ/discrete actions not yet needed.

## Open directions

- Decouple the two energies: a *learned projection head* on top of the
  (content-rich) state that is straightened/monotone, so raw-geometry
  planning and value decodability stop competing for the same metric.
- Edit track: break energy ties among near-duplicate edits (margin/ranking
  loss over candidate edits from the same state — distillation option (c));
  the goal-distance energy already beats the value head there.
- Use the trained hierarchy (macro-actions + F_hi) in the planner: propose
  K-step macro rollouts, refine with the low-level model (HWM-style).
- Harder worlds: deeper DAGs, larger moduli, compositional templates,
  natural-language surface variation; scale d_model/params.
- Variational action prior p(a | s, g) for sampling actions without
  enumerating the interface (open-ended generation).
