# Results — first experimental cycle (2026-07-09)

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
| **disc_valgrad** | **0.665** | **0.835** | **0.920** | **0.07** (look-2) |

Edit track: edit_base 0.115 → **edit_valgrad 0.425** (random 0.06).

## The five findings

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

## Stability

No collapse in any run: VICReg + EMA targets held per-dim std ≈ 1.0 and
effective rank 220–243/256 throughout; FSQ/discrete actions not yet needed.

## Open directions

- Edit track still trails discourse: add a frozen-anchor outcome objective
  for buffers (anchor the *edited sentence* embedding), and diagnose the
  planner's slack-insensitivity (it doesn't convert extra budget into
  fixes — likely value ties between no-op replaces and true fixes).
- Use the trained hierarchy (macro-actions + F_hi) in the planner: propose
  K-step macro rollouts, refine with the low-level model (HWM-style).
- Harder worlds: deeper DAGs, larger moduli, compositional templates,
  natural-language surface variation; scale d_model/params.
- Variational action prior p(a | s, g) for sampling actions without
  enumerating the interface (open-ended generation).
