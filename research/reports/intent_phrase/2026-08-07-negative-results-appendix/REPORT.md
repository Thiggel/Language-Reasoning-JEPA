# Negative-results appendix: what was measured and rejected on the way to the frozen recipe

_2026-08-07. Compilation report; no new experiments. Every number below is
copied from a dated report or a run-directory metrics file, cited per item.
Companion LaTeX table: `appendix_negative_results.tex` in this directory._

## Shared evaluation protocol (applies to every row unless stated)

All results are **strict success** (solve in exactly the minimum number of
valid actions) on 300 validation episodes of stylized iGSM, root-balanced beam
search with width 8, planning depths 1/2/4/8/16. Depth 1 uses the current
symbolic feasible menu; depth > 1 additionally uses symbolic future feasible
menus and is therefore **candidate-privileged** — exactly the same protocol as
the paper headline, so every comparison below is like-for-like. One seed per
cell unless a seed count is given.

**Reference (frozen recipe, 5 seeds)**: horizon-blind endpoint Energy
`E(root, imagined endpoint, z_0)`, logistic pairwise ranking of recursively
imagined endpoints, training horizons {1,2,4,8}, 0.25 root pair-difference
auxiliary: strict **.126 / .450 / .837 / .879 / .884** at depths 1/2/4/8/16
(slack-2 .541/.863/.979/.991/.995). Floors/ceiling on the same episodes:
random .053, first-feasible .247, oracle encoded-goal distance 1.000
(**oracle** — uses the symbolically encoded solved state at plan time).
Source: `../2026-08-06-coherent-endpoint-energy/REPORT.md`.

**The contrast axis these negatives establish**: supervision that only orders
imagined endpoints relative to each other (pairwise ranking) survives
recursive imagination; supervision that pins the score to an absolute or
bootstrapped calibration target (MSE distance, TD, expectile value) does not —
the calibration is learned on real states and breaks on recursively imagined
ones, so deeper search stops helping or actively hurts.

---

## Family (i): absolute / value-calibrated energies

### i.1 State–goal Energy MSE regression

- **What it does**: trains the Energy head to regress the absolute latent
  distance between a state and the goal representation, instead of only
  ranking candidates.
- **Hypothesis**: an absolutely calibrated distance-to-goal score should be
  directly comparable across states and depths, making it a natural terminal
  beam score.
- **Result**: one-seed terminal rows D1 .143, D4 .117, D8 .113, D16 .117
  (base LR 3e-4; joint/dense/detached rollout variants .067–.203 at depth). A
  matched **five-seed** evaluation of the existing state-Energy head gives
  .407 / .125 / .118 / .118 at D1/4/8/16 — strong on real current states,
  collapsed on imagined endpoints, ruling out a seed-specific explanation.
- **Verdict**: rejected. Accurate absolute regression is hard, and a head
  calibrated on real states does not transfer to recursively imagined ones.
- **Evidence**: candidate-privileged at D>1, same as headline.
- **Source**: `../2026-08-06-comprehensive-project-state/REPORT.md`
  (sections 9, 9a "Direct advantage and state-Energy targets",
  "Five-seed existing state-Energy head").

### i.2 Advantage-difference regression (cumulative composition)

- **What it does**: regresses per-action advantage differences and scores a
  candidate sequence by summing predicted per-edge changes along the path.
- **Hypothesis**: if each action's contribution is calibrated, contributions
  should add up over an imagined sequence.
- **Result**: D1 .117–.157 across rollout treatments; D4–D16 .027–.090. The
  best cumulative variant (listwise rank on the difference target) reached
  only .190 at D16.
- **Verdict**: rejected. Calibration error accumulates under summation, and an
  advantage learned on one state distribution is not additive under
  recursively imagined states.
- **Evidence**: candidate-privileged at D>1.
- **Source**: `../2026-08-06-comprehensive-project-state/REPORT.md`
  (section 9a tables).

### i.3 Ranking plus MSE (mixed objective)

- **What it does**: adds an absolute MSE calibration term on top of the
  pairwise ranking loss on the same head.
- **Hypothesis**: calibration and ordering could be complementary — ranking
  fixes the order, MSE fixes the scale.
- **Result**: in the direct-difference cells, ranking+MSE fell to .04–.05 at
  D4–D16 (versus .837–.884 for the recipe); in the 2026-08-06 sparse
  factorial the pure-MSE column never exceeded .283 at any depth and was
  .053–.190 at D16. In the older global-beam localization pilots (historical,
  pre-repair evaluator — localization only, not a paper comparison), MSE
  weights .1/.5/1.0 gave D4 .040/.035/.020 versus .190 for ranking-only.
- **Verdict**: rejected. The MSE term does not merely fail to help — it can
  overwhelm the useful ordering signal.
- **Evidence**: candidate-privileged at D>1; the global-beam pilot rows are
  additionally labeled historical (superseded evaluator).
- **Source**: `../2026-08-06-comprehensive-project-state/REPORT.md`
  (sections 9, 9a, "August 4 global-beam localization");
  `runs/autonomy/intent_phrase/2026-08-06-intent-sparse-multidepth-energy-factorial-v3/*/mse/terminal_depth*_slack0.json`.

### i.4 TD-Q (SARSA, amortized value) as the planning energy

- **What it does**: trains a Q-head with bootstrapped SARSA temporal-difference
  targets on offline traces and scores candidates by the amortized value —
  no test-time imagination needed.
- **Hypothesis**: a bootstrapped value should summarize the future without
  paying for recursive rollout, the standard RL alternative to search.
- **Result**: .190 / .210 / .240 / .223 / .220 at D1–D16. Myopically stronger
  than the recipe at D1 (.190 vs .126) but flat with depth; slack-2
  *decreases* with depth (.617 at D1 to .470 deep) — deeper search actively
  hurts it.
- **Verdict**: rejected as the planning energy. The bootstrapped calibration
  cannot exploit recursive imagination; kept as the honest simple-TD baseline
  row in the paper.
- **Evidence**: candidate-privileged at D>1, identical protocol.
- **Source**: `../2026-08-06-competitor-energy-baselines/REPORT.md` ("First
  results"); round `2026-08-06-intent-td-baseline-cells-v1`.

### i.5 Expectile goal-value (Destrade-style V(s,g) = -||E(s)-E(g)||)

- **What it does**: trains a distance-shaped value with IQL-style expectile TD
  regression and plans by minimizing the imagined endpoint's learned distance
  value.
- **Hypothesis**: expectile TD is the literature's robust way to fit sparse
  offline values; a distance-shaped value should transfer to endpoint scoring.
- **Result**: .057 / .063 / .103 / .173 / .187 at D1–D16 — barely above the
  random floor (.053) at shallow depth and never above .19.
- **Verdict**: rejected; near random. Same failure mode as i.4 in a
  distance-shaped parameterization.
- **Evidence**: candidate-privileged at D>1.
- **Source**: `../2026-08-06-competitor-energy-baselines/REPORT.md`; round
  `2026-08-06-intent-td-baseline-cells-v1`.

### i.6 Expectile-TD shaping auxiliary on the frozen recipe

- **What it does**: keeps the full frozen ranking recipe and merely adds an
  expectile-TD value loss as a shaping auxiliary on the same backbone.
- **Hypothesis**: even if TD fails alone, its bootstrapped signal might inject
  useful long-horizon information into the ranking energy.
- **Result**: .150 / .323 / .847 / .863 / .863 — deep depths within one-seed
  noise of the recipe (.837/.879/.884), and a clear regression at D2
  (.323 vs .450).
- **Verdict**: rejected. TD-as-ingredient buys nothing and damages the D2
  regime; consistent with the pure TD baselines' failure to exploit search.
- **Evidence**: candidate-privileged at D>1.
- **Source**: `../2026-08-06-coherent-endpoint-energy/REPORT.md`
  ("Frozen-recipe variants"); round
  `2026-08-07-intent-frozen-recipe-variants-v1`.

### i.7 Faithful TD-JEPA (successor features + task embedding)

- **What it does**: the faithful adaptation of TD-JEPA (arXiv:2510.00739) — a
  successor-feature predictor T(phi(s), u(a), z_task) trained with a
  bootstrapped TD loss, task embedding regressed from z_0, scored at plan
  time by T·z_r with z_r ridge-fit to the sparse task reward.
- **Hypothesis**: the literature's zero-shot TD value parameterization might
  succeed where our simpler SARSA head failed.
- **Result (2026-08-07, one seed)**: strict .133 / .130 / .117 / .107 / .107
  at D1–D16; slack-2 falls with depth (.530 at D1 to .327 at D16). No depth
  scaling; deeper search slightly hurts, like TD-Q.
- **Verdict**: rejected as a planning energy; kept as the faithful TD-family
  baseline row. Confirms i.4 was not a strawman: the failure is the
  bootstrapped calibration, not the parameterization.
- **Evidence**: candidate-privileged at D>1, identical protocol.
- **Source**:
  `runs/autonomy/intent_phrase/2026-08-07-intent-faithful-baselines-v1/baseline-td-jepa-s0-v1/terminal_depth*_slackcurve.json`
  (strict = slack-0 entry); adaptation contract in
  `../2026-08-06-competitor-energy-baselines/REPORT.md`.

### i.8 Takai-style GoalHead (predicted goal latent + distance planning)

- **What it does**: trains g(z_0) to predict the solved-state latent
  (MSE + cosine against the EMA-encoded solved endpoint, training only) and
  plans by minimizing the imagined endpoint's distance to that predicted goal
  — the non-oracle counterpart of the oracle encoded-goal diagnostic (1.000).
- **Hypothesis**: if a predicted goal latent is accurate enough, plain latent
  distance should inherit the oracle diagnostic's perfect deep-search
  behavior without symbolic goal access.
- **Result (2026-08-07, one seed)**: strict .137 / .037 / .057 / .060 / .060
  at D1–D16; slack-2 .610 at D1 collapsing to .350 at D2. Roughly matches
  the recipe at depth 1, collapses as soon as search deepens.
- **Verdict**: rejected. An absolute distance to an imperfectly predicted
  goal point is exactly the calibration that recursive imagination breaks;
  the gap to the oracle version (1.000) shows the goal prediction, not the
  distance planner, is the bottleneck.
- **Evidence**: candidate-privileged at D>1; the contrasted 1.000 ceiling is
  oracle.
- **Source**:
  `runs/autonomy/intent_phrase/2026-08-07-intent-faithful-baselines-v1/baseline-goal-head-s0-v1/terminal_depth*_slackcurve.json`;
  adaptation contract in
  `../2026-08-06-competitor-energy-baselines/REPORT.md`.

---

## Family (ii): supervision-structure variants

### ii.1 Sparse multidepth prefix supervision (2026-08-06 factorial)

- **What it does**: supervises the Energy on root-to-endpoint prefixes of many
  depths {0,1,2,4,8,16} from the same root, instead of only terminal
  endpoints of sampled rollout horizons.
- **Hypothesis**: seeing every prefix depth during training should give the
  head an in-support score at every beam-pruning depth.
- **Result**: every completed factorial cell (energy loss ∈ {MSE, rank, both}
  × geometry ∈ {none, straightening, projected straightening, monotonicity,
  projected monotonicity} × {energy-only, dense dynamics}, batch 16) landed
  far below the plain recipe: best cell overall was dense-dynamics /
  no-geometry / rank+MSE at D16 = **.550**; most cells sit at .1–.3 with
  non-monotone depth curves (e.g. energy-only/none/rank
  .113/.083/.187/.213/.373 at D1–D16).
- **Verdict**: rejected. Spreading the supervision over prefixes dilutes the
  endpoint-ranking signal that actually drives deep search; ~30 points below
  the recipe at D16.
- **Evidence**: candidate-privileged at D>1.
- **Source**: `../2026-08-06-multidepth-energy-factorial/REPORT.md` (design),
  `../2026-08-06-coherent-endpoint-energy/REPORT.md` (finding 5), exact
  per-cell numbers in
  `runs/autonomy/intent_phrase/2026-08-06-intent-sparse-multidepth-energy-factorial-v3/*/{mse,rank,both}/terminal_depth*_slack0.json`.

### ii.2 Dense within-horizon endpoint supervision

- **What it does**: applies the endpoint-ranking loss at intermediate rollout
  steps too, comparing candidates only within the same horizon (the repaired,
  valid version; an earlier cross-horizon version was invalid and discarded).
- **Hypothesis**: dense targets should stabilize the recursive predictor and
  give the head more supervision per example.
- **Result**: H2/H4/H8/H16 dense models reached only .080/.183/.063/.053 at
  D16 (D4 .087–.183); earlier local-GAR rollouts with dense losses likewise
  dropped to .080–.087 at depth versus .210–.223 for the same recipe without
  the dense term.
- **Verdict**: rejected. One recursively shared predictor is asked to match
  every intermediate EMA state while the endpoint objective wants the full
  composition to preserve ordering; the updates conflict (interpretation —
  gradient-conflict measurements were not completed).
- **Evidence**: candidate-privileged at D>1.
- **Source**: `../2026-08-06-comprehensive-project-state/REPORT.md`
  (section 8); round
  `2026-08-06-intent-within-horizon-dense-endpoint-repair-v1`.

### ii.3 Fixed training horizon H=4

- **What it does**: trains all endpoint-ranking examples at exactly horizon 4
  instead of mixing horizons {1,2,4,8}.
- **Hypothesis**: a single clean horizon gives one coherent endpoint semantics
  and was the strongest single-horizon cell at its own depth (.910 at D4).
- **Result**: at the untrained depth 2 it scores **.290** versus .450 for the
  mixed-horizon recipe — the head is incoherent off its training support.
  (Fixed H8 and H16 are worse everywhere: D4 .650 and .503.)
- **Verdict**: rejected as the paper method. Peak performance at one depth
  does not transfer across query depths; horizon mixing is what buys the
  transfer.
- **Evidence**: candidate-privileged at D>1.
- **Source**: `../2026-08-06-coherent-endpoint-energy/REPORT.md` (finding 4),
  `../2026-08-06-comprehensive-project-state/REPORT.md` (section 7); round
  `2026-08-06-intent-depth2-reference-evals-v1`.

### ii.4 Adding H=16 rollouts to the training mix

- **What it does**: extends the training horizon set from {1,2,4,8} to
  {1,2,4,8,16} so the deepest query depth is in-support.
- **Hypothesis**: training on H16 rollouts should help exactly at query depth
  16.
- **Result**: under the horizon-blind head with the 0.25 auxiliary, horizons
  {1,2,4,8} give .827/.870/.875 at D4/8/16 (2 seeds) versus .808/.832/.833
  for {1,2,4,8,16} (3 seeds) — the H16 set is *worse*, including at D16.
- **Verdict**: rejected. H16 rollouts dilute the fixed training budget with
  hard, drift-prone examples; the horizon-blind Energy already generalizes to
  query depth 16 without ever training on H16.
- **Evidence**: candidate-privileged at D>1.
- **Source**: `../2026-08-06-coherent-endpoint-energy/REPORT.md`
  ("Recipe freeze").

### ii.5 Horizon input to the Energy head

- **What it does**: feeds the query depth (as `log(1+H)/4`) into the Energy
  head so the score can condition on how far ahead it is looking.
- **Hypothesis**: different depths might need different calibrations, so the
  head should know the horizon.
- **Result**: horizon-blind five-seed .123/.438/.777/.800/.801 versus the
  horizon-conditioned twin .135/.447/.780/.820/.818 (2 seeds) — overlapping
  at every depth; the frozen horizon-blind recipe (.126/.450/.837/.879/.884,
  5 seeds) then matched or exceeded the historical horizon-conditioned
  headline (.834/.874/.877 at D4/8/16) everywhere.
- **Verdict**: dropped as unnecessary (a simplification win, not a failure).
  With no depth conditioning there is nothing to be out-of-support of, which
  dissolves the earlier semantics objection.
- **Evidence**: candidate-privileged at D>1.
- **Source**: `../2026-08-06-coherent-endpoint-energy/REPORT.md`
  (findings 1–2, five-seed promotion); round
  `2026-08-06-intent-coherent-endpoint-energy-v1`.

---

## Family (iii): geometry regularizers

All four regularizers were measured inside the 2026-08-06 sparse-prefix
factorial (ii.1), energy-only arm, so their absolute levels carry that
formulation's handicap; the informative comparison is against the matched
no-geometry cells in the same factorial. Numbers below are strict success at
D1/2/4/8/16 from
`runs/autonomy/intent_phrase/2026-08-06-intent-sparse-multidepth-energy-factorial-v3`.

- **What they do**: temporal straightening penalizes curvature of the latent
  rollout trajectory (raw = on latent states, projected = in a learned
  projection); monotonicity pushes the Energy (raw) or projected distance to
  decrease along teacher-confirmed improving steps.
- **Hypothesis**: straighter or monotone latent trajectories should make
  recursively imagined endpoints easier to rank.
- **Results (rank-loss column; matched no-geometry control first)**:

| Geometry (energy-only, rank loss) | D1 | D2 | D4 | D8 | D16 |
|---|---:|---:|---:|---:|---:|
| none (control) | .113 | .083 | .187 | .213 | .373 |
| straightening (raw) | .107 | .107 | .187 | .167 | .187 |
| straightening (projected) | .123 | .100 | .183 | .183 | .290 |
| monotonicity (raw) | .093 | .093 | .177 | .083 | .273 |
| monotonicity (projected) | .117 | .087 | .207 | .147 | .300 |

  In the rank column no regularizer beats its control at D16; the single best
  geometry cell anywhere in the factorial (projected straightening, rank+MSE)
  reaches .493 at D16, still below the best no-geometry cell (.550) and ~.39
  below the recipe (.884).
- **Verdict**: all four rejected — no consistent benefit in combination with
  ranking, several cells clearly harmful.
- **Evidence**: candidate-privileged at D>1. Caveat: measured only inside the
  sparse-prefix formulation; a geometry regularizer on top of the frozen
  recipe itself was never run (see "Unverified" below).
- **Source**: `../2026-08-06-coherent-endpoint-energy/REPORT.md` (finding 5);
  per-cell JSONs as cited above.

---

## Family (iv): architecture

### iv.1 Non-residual (direct) MLP prediction

- **What it does**: the recursive predictor outputs the next latent state
  directly instead of a residual update added to the current state.
- **Hypothesis**: the residual parameterization might be load-bearing for
  stable recursive rollouts.
- **Result**: .120 / .410 / .813 / .850 / .860 — slightly below the five-seed
  recipe (.126/.450/.837/.879/.884) at every depth but within about one seed
  standard deviation.
- **Verdict**: roughly neutral; residual is kept but labeled second-order.
  The first-order architectural choice is the predictor class, not the
  residual connection.
- **Evidence**: candidate-privileged at D>1.
- **Source**: `../2026-08-06-coherent-endpoint-energy/REPORT.md`
  ("Frozen-recipe variants"); round
  `2026-08-07-intent-frozen-recipe-variants-v1`.

### iv.2 Causal-transformer predictor (status, not a completed negative)

- **What it does**: replaces the shared MLP predictor with a causal
  transformer over the action sequence.
- **Hypothesis**: a sequence model should compound imagined steps with less
  drift than a recursively applied MLP.
- **Result so far**: in the earlier local-GAR setting (five seeds, current
  feasible menu, D1) the transformer predictor scored .366 ± .047 versus
  .433 ± .019 for the MLP — no improvement. The transformer under the
  **endpoint-Energy recipe has not been trained**: it needs training-time
  recursive imagination, which a code guard currently blocks.
- **Verdict**: open, not a measured negative for the frozen recipe; listed
  here so the appendix does not overclaim.
- **Evidence**: the .366 figure is the information-matched D1 interface (not
  candidate-privileged); the endpoint-recipe cell is pending.
- **Source**: `../2026-08-06-comprehensive-project-state/REPORT.md`
  (section 1); `../../../projects/intent_phrase/CAMPAIGN_LOG.md`
  (next steps, item 6).

---

## Unverified — needs rerun (no findable numbers; do not cite as measured)

1. **Dense-dynamics × geometry factorial cells** (straightening /
   monotonicity, raw and projected, combined with dense latent dynamics):
   submitted to Lise
   (`runs/autonomy/intent_phrase/2026-08-06-intent-sparse-multidepth-energy-factorial-lise-remainder-v4`,
   `-v5`) but no result JSONs are present on this filesystem (Lise storage is
   separate; nothing was synced back). The energy-only geometry arm plus the
   dense no-geometry control suffice for the rejection in (iii); these cells
   could only close a ~.33 gap to the recipe, which the 2026-08-06 report
   judged implausible, but that judgment is not a measurement.
2. **Geometry regularizers on top of the frozen recipe** (rather than inside
   the sparse-prefix factorial): never run.
3. **Causal-transformer endpoint-Energy cell**: blocked by the training-time
   recursive-imagination guard (iv.2); planned, not run.

## Summary table (one row per variant)

Recipe reference at the comparison depth is the five-seed frozen-recipe value
(.126 D1, .450 D2, .837 D4, .884 D16). "CP" = candidate-privileged at D>1
(same as headline); seeds = 1 unless noted.

| Variant | Family | Key number (strict) | Recipe, same depth | Verdict |
|---|---|---|---|---|
| State–goal Energy MSE regression | absolute | D16 .117 (5-seed head: .118) | .884 | rejected: absolute calibration collapses on imagined endpoints |
| Advantage-difference regression (cumulative) | absolute | D16 .027–.090 | .884 | rejected: errors accumulate under summation |
| Ranking + MSE mixed objective | absolute | D4–16 .04–.05 (direct-diff cells) | .837–.884 | rejected: MSE swamps the ordering signal |
| TD-Q (SARSA, amortized) | absolute | D1 .190, D16 .220; slack-2 falls .617→.470 | D1 .126, D16 .884 | rejected: stronger myopically, cannot exploit search |
| Expectile goal-value | absolute | D16 .187 (random .053) | .884 | rejected: near random |
| Expectile-TD shaping auxiliary | absolute | D2 .323, D16 .863 | .450, .884 | rejected: no gain, D2 regression |
| Faithful TD-JEPA (successor features) | absolute | D1 .133, D16 .107; slack-2 falls .530→.327 | D1 .126, D16 .884 | rejected: no depth scaling, deeper search hurts |
| Takai-style GoalHead (predicted-goal distance) | absolute | D1 .137, D2 .037, D16 .060 | D1 .126, D16 .884 | rejected: collapses beyond depth 1 (oracle-goal ceiling 1.000) |
| Sparse multidepth prefix supervision | supervision | best cell D16 .550; most cells .1–.3 | .884 | rejected: dilutes endpoint-ranking signal |
| Dense within-horizon endpoint loss | supervision | D16 .053–.183 | .884 | rejected: dense targets conflict with endpoint ordering |
| Fixed training horizon H4 | supervision | D2 .290 (D4 .910) | D2 .450 | rejected: incoherent off training support |
| H16 in training horizon mix | supervision | D16 .833 (3 seeds) | .875 (2 seeds, matched ablation) | rejected: budget dilution, no D16 benefit |
| Horizon input to Energy head | supervision | 5-seed D16 .801 vs twin .818 (2 seeds), overlapping | — | dropped: unnecessary, simplification win |
| Straightening (raw) | geometry | D16 .187 (control .373, same factorial) | .884 | rejected: no benefit, often harmful |
| Straightening (projected) | geometry | D16 .290 rank / .493 rank+MSE | .884 | rejected: no consistent benefit |
| Monotonicity (raw) | geometry | D16 .273 (control .373) | .884 | rejected: no benefit |
| Monotonicity (projected) | geometry | D16 .300 (control .373) | .884 | rejected: no benefit |
| Non-residual MLP prediction | architecture | D16 .860 | .884 ± .023 | neutral (~1 SD below); residual kept, second-order |
| Causal-transformer predictor | architecture | GAR D1 .366 ± .047 vs MLP .433 ± .019 (5 seeds); endpoint recipe untested | — | open: pending, not a measured negative |

All rows share the headline's candidate-privileged evaluation at depth > 1;
the only oracle number cited anywhere above is the encoded-goal ceiling
(1.000), labeled as such.
