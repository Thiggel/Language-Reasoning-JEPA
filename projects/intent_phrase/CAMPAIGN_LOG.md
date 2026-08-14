# Intent-phrase campaign log (living handoff)

_Keep short. Compress completed stages into a few lines; details live in
`research/reports/intent_phrase/<date>-*/REPORT.md` and are mirrored to
`/vol/home-vol2/ml/laitenbf/TextJEPA-paper/reports/`. Last update: 2026-08-14._

## CURRENT STATE (2026-08-14) — read this first

- **Recipe (frozen)**: `mix4_aux025_nohorizon` + LDAD
  (`model.observed_action_ldad=true objective.observed_action_ldad.weight=1.0`),
  EMA+stopgrad+VICReg, endpoint-Energy ranking, GPT2-small budget 157M, bs 8.
- **Eval protocol (frozen 2026-08-14, snapshot 8acd62c)**: band problems on
  SOLUTION LENGTH via `necessary_range` (ID nec 8-15 @ caps
  max_op=32/max_edge=40/op_range=[3,32]; OOD nec 20-28 @ caps 48/64/[3,48]);
  proportional attempt budget `slack_frac` (budget = necessary + ceil(frac ×
  necessary)); attempted-mask scoped to current state, resets on progress.
  Paper instrument = success vs attempt budget in multiples of necessary.
  Details: `research/reports/intent_phrase/2026-08-14-protocol-fix/REPORT.md`.
- **Interface decision (owner, 2026-08-14)**: `feasible_menu` DEMOTED to
  appendix control — structurally unsalvageable (menus intrinsically ~2-3
  wide, 41-78% necessary, wasted picks unrepeatable → random ~.7-.98).
  MAIN results = menu-free interfaces: `full_catalogue`, `codebook_ground`
  (nearest training-catalogue action), `ldad_cycle`, `autonomous`.
- **Headline finding so far**: menu-trained Energy has NO feasibility signal
  menu-free — below random on full_catalogue (planner .530/.140 vs random
  .700/.480 ID/OOD at 5× budget, invalid-proposal rate .79-.88). Narrative:
  menu-trained energy fails menu-free → cycle-consistency (ldad_cycle,
  stylized AUC .94) recovers feasibility → autonomous operation.
- **Running now**: `2026-08-13-igsm-hard-base-v1` LR screen 12/18 COMPLETED,
  6 RUNNING (tok-lm 1e3/1e4/3e3, ldad-3e3, sent-lat-1e4, tdjepa-1e4) on
  gruenau via `scripts/gruenau_dispatcher.sh` (launch detection reads cell
  `state` file; never pkill by pattern on gruenau1 — kill by PID).
  Cells' built-in slackcurve evals are OLD feasible-menu = sanity only
  (random .98); do not pick winners from them.
- **Next (critical path)**: #28 port `ldad_cycle`+`codebook_ground` to
  `FaithfulPlanner` and eval on the fixed bands vs the .700/.480 random
  reference; #29 re-run 12 void fullcat OOD rows from post-8acd62c snapshot;
  #30 pick budget multiples {1,1.5,2,3,5}×; #31 verify recipe wiring
  (geo_rank=0 but geo_horizon_rank .554 active despite "nohorizon" name).
  Open question: checkpoints trained on op_range [3,21] while eval bands
  widened — retraining may be needed for mains.

## RECIPE UPDATE (2026-08-08): LDAD added to the recipe (new headline)

Out of the owner's stabilizer sweep. New headline recipe =
`mix4_aux025_nohorizon` + `model.observed_action_ldad=true,
objective.observed_action_ldad.weight=1` (EMA+stopgrad+VICReg kept).
FIVE SEEDS (strict D1..D16): .202+-.049 / .729+-.026 / .885+-.026 /
.965+-.015 / .975+-.012 — dominates the 2026-08-07 recipe
(.126/.450/.837/.879/.884) at EVERY depth; D2 +.28 fixes the known
shallow-depth weakness (theory T4: best-of-R label noisiest at H=1;
LDAD's action-displacement decoding sharpens one-step geometry).
Frozen-protocol check PASSED (1 seed): head distilled from the FROZEN
LDAD backbone .183/.763/.887/.940/.940 — the gain is in the geometry.
Runner-up: SIGReg replacing VICReg (5 seeds .143/.434/.888/.959/.959,
frozen check .953 D16) — deep-depth gain only. SIGReg+LDAD without
VICReg does NOT compose (.843 D16); three-way combo running.

## Frozen method (2026-08-07, superseded above for the headline)

Horizon-blind endpoint Energy `E(root, imagined endpoint, z_0)` — no depth
input — logistic pairwise ranking of recursively imagined endpoints, labels =
EMA-latent distance of true rollout outcomes to the encoded solved state
(goal used only in training labels). Training horizons {1,2,4,8}, R=4
rollouts, K=2 alternatives, 0.25 root pair-difference auxiliary, no dense
loss. Cell variant `mix4_aux025_nohorizon`, snapshot `ee49091`.

**Five-seed headline (300 val episodes, root-balanced beam B=8, feasible menu
at D1, symbolic future menus [candidate-privileged] at D>1):**
strict .126/.450/.837/.879/.884 at depths 1/2/4/8/16
(slack-2 .541/.863/.979/.991/.995; random .053, first-feasible .247, oracle 1.0).

## Decided (details in dated reports)

- Horizon input to the Energy head: unnecessary (5 seeds). H16 training
  rollouts: harmful dilution. Fixed-H4: incoherent off-support (.29 at D2).
- Sparse prefix supervision, straightening, monotonicity (raw + projected):
  rejected (2026-08-06 factorial, all cells far below plain endpoint recipe).
- TD-Q (SARSA) baseline: .19 D1, no depth scaling. Expectile goal-value:
  near random. TD-as-shaping-auxiliary: no gain, D2 regression.
- Non-residual MLP prediction: roughly neutral (~1 SD below). Residual choice
  is second-order; predictor class (MLP vs causal transformer) is first-order.
- Repo: researchctl removed 2026-08-06; paper lives in
  `/vol/home-vol2/ml/laitenbf/TextJEPA-paper/` (own git).

## Results 2026-08-07 (one seed each unless noted; strict success D1..D16)

- Faithful TD-JEPA (arXiv:2510.00739): .133/.130/.117/.107/.107 — declines
  with depth. Takai GoalHead (JSAI 2026): .137/.037/.057/.060/.060 —
  collapses after D1. Same family signature as TD-Q/expectile: absolute
  calibrations don't survive recursive imagination. Details in the
  competitor-baselines report.
- Frozen-backbone triangle COMPLETE (1 seed each, identical frozen
  distillation protocol, strict D1..D16):
  plain JEPA-only backbone .100/.257/.643/.773/.783;
  TD-shaped backbone (td_q gradients, no ranking) .153/.360/.907/.927/.913;
  ranking-shaped backbone .137/.433/.830/.840/.837;
  joint-training reference (5 seeds) .126/.450/.837/.879/.884.
  Reading: (a) post-hoc distillation works on ALL three — ordinal
  structure largely emerges from JEPA objectives alone (plain .78 at D16
  vs random .05); (b) value-gradient shaping of the backbone adds ~.1
  (plain vs TD is the clean pair — same cheap data settings); (c) TD is a
  good geometry SHAPER but a bad SCORE (TD-JEPA/TD-Q planners fail with
  depth, yet TD-shaped backbone + ranking head is the best cell so far) —
  vindicates the owner's "TD as an ingredient" idea in modified form.
  Caveat: ranking-shaped backbone saw richer data (H mix, R=4) than
  plain/TD (H=1, R=1), so (b) uses plain-vs-TD only. Triangle COMPLETE at
  3 SEEDS (strict D16 mean+-sd): plain .786+-.015 < ranking-shaped
  .829+-.009 < TD-shaped .902+-.017 — all pairwise gaps non-overlapping;
  the ~.12 plain-vs-TD value-shaping gain at depth is the paper claim.
  Note: each run dir holds TWO eval sets — the paper-relevant pure
  endpoint-Energy planner is under `horizon_only/`; the root-level
  slackcurve files are a hybrid local-pruning variant (worse deep) —
  always collect from `horizon_only/`.
- Counterfactual K sweep COMPLETE (seed 0, D2 strict): K=0 .237, K=1 .410,
  K=2 .450 (5 seeds), K=3 .323, K=8 .313, K=16 .407, K=64 .407. Deep depths
  all ~.85-.93. Two caveats found: (a) K saturates at catalogue size
  (n_vars <= 12, alts = others[:K]) — K=16 and K=64 are bit-identical;
  report the axis as K in {0,1,2,3,8,all}. (b) K=0/8/32 ran from the
  pre-merge snapshot, K=1/3/16/64 post-merge; the merged baseline heads
  shift RNG at init, so same-seed cross-snapshot cells are NOT directly
  comparable — treat cross-snapshot differences as containing an extra
  seed-sized noise term. Robust conclusion: K=0 clearly hurts shallow
  depths; any K >= 1
  is equivalent within single-seed noise. Recipe K=2 stands; exact K is
  second-order. K=0 s1 replicates s0 (D2 .263 vs .237; deep ~.88) — the
  K=0 shallow penalty is now 2-seed. Full-curve seeds only if the paper
  wants error bars on every K.
- Theory probes (audit_theory_predictions.py, mix4 s0): T2 monotone
  invariance exact (planner metrics bit-identical under exp/cube/affine).
  T1 Kendall tau geometry-vs-steps-to-go only .46 mean — raw geometry is
  moderately ordinal; ranking head adds real information. T4: best-of-R
  label agrees with exact optimal root ordering .48 at H=1 but .83+ at
  H>=2 — quantifies why depth-1 is weak and lookahead scales. Registered
  in the theory report before the pure-backbone cell returns.
- Negative-results appendix compiled (17-row table + LaTeX):
  `research/reports/intent_phrase/2026-08-07-negative-results-appendix/`.
- Repo: full test suite green again (703 passed; consolidation breakage in
  token_igsm packed attention + samplers restored).

## Running / queued (2026-08-07)

- Counterfactual K in {3, 16, 64} still training (round
  `2026-08-07-intent-counterfactual-scaling-v1`, snapshot `9ca28ee`).
- Full-catalogue recipe TRAINING cell `mix4-fullcat-s0-v1` (gruenau10 GPU0,
  snapshot `b788372`): ranking alternatives from the full catalogue,
  inherited eval also plans full-catalogue at every depth (no feasibility
  oracle anywhere).
- Plain-backbone chain (gruenau1 RTX 6000): `plain-backbone-s0-v1` trains
  backbone objectives only (no value/TD/ranking gradients), then
  `frozen-backbone-plain-s0-v1` distills the Energy head on it frozen.
- Interface controls RESULTS (2026-08-08): WITHOUT the feasible menu every
  method collapses to ~0 strict success at every depth (invalid-action
  rates 73-100%, invalid=noop): recipe s0 evaluated full-catalogue;
  TD-JEPA and GoalHead evaluated full-catalogue; recipe RETRAINED with
  full-catalogue ranking candidates; recipe + flat Gaussian action prior
  (learned_catalogue, top-8/top-3/top-2, with and without feasibility
  gate). Collapse is uniform — no method is privileged by the menu
  relative to another. Diagnosis: the prior has signal (correct action
  mean rank 3.6 of ~9 catalogue entries) but the PAIRWISE feasibility head
  is uninformative (58% acc, near-constant logits) because feasibility in
  iGSM is relational (all prerequisites executed). Fix committed
  (`1321619`): history-attention feasibility head wired into the intent
  track (causal executed-action history in training + planner gating);
  cell `mix4-prior-hist-s0-v1` came back collapsed for a DIFFERENT reason:
  the recipe experiment sets value_detach=false, so the prior NLL
  collapsed the action encoder (action_std 2.4 -> 0.002 in one epoch).
  Fixed (`d2f1cf5`: prior/feasibility supervision always detaches inputs);
  rerun `mix4-prior-hist-s0-v2` had HEALTHY embeddings (action_std 7.5)
  and STILL fails menu-free (~0 strict, invalid 72-82%, feasibility acc
  52% = chance, prior rank ~4.3/9). CONCLUSION (3 training iterations):
  menu-free planning fails for every method because the pooled state does
  not expose per-candidate prerequisite structure (it lives in the prompt
  text; nothing in JEPA training forces it into the state). This is the
  paper's honest interface-axis story: headline WITH menu (fair — uniform
  collapse without it for all methods), menu-free as a documented
  limitation + the collapse control. LM free-generation baselines
  predictably consistent; skip unless a reviewer asks.
  CLOSED 2026-08-10 after two final escalations on the LDAD backbone:
  Gaussian-NLL prior (rank ~5, menu-free ~0) and CONTRASTIVE prior
  (catalogue-softmax CE trained on exactly the plan-time ranking; ends AT
  the chance floor 2.57, menu-free .000-.030). No further menu-free work
  via STATE-readout heads; see interface-controls report.
  CORRECTED 2026-08-11 (state-readout-controls report): the information IS
  in the state — oracle probes decode feasibility from (s_t,u) at AUC .935
  (resolvedness .982), and the same head trained offline lands 0.47 nats
  BELOW chance. The failure is the next-action-imitation objective (rewards
  preference, barely rewards the conjunctive parent-lookup), not missing
  information. Say "imitation-trained state readouts stay near chance;
  cycle-consistency recovers the same information through the trained
  dynamics" — NOT "the state lacks the information".
  REOPENED 2026-08-10 via the DYNAMICS: LDAD cycle-consistency — score
  each catalogue candidate by how well the LDAD displacement decoder
  reconstructs the candidate's own phrase from the predictor's imagined
  displacement (feasible/infeasible AUC .94, no oracle, no training),
  mask the planner's own executed actions. First real menu-free numbers
  (`2026-08-10-intent-ldad-cycle-v1`, LDAD checkpoint, top-2): slack-4
  .25-.31 at ALL depths, invalid 24-43% (all state-readout attempts: ~0).
  Top-3 eval running. Owner position 2026-08-10: catalogue enumeration
  (variable names, prompt-derivable) is acceptable for the paper — the
  privileged part was feasibility, now learned. OPTIONAL STRETCH: fully
  open-ended proposals via a state-conditioned intent-phrase generator
  head vetted by cycle-consistency; would let LM baselines run pure
  free-generation (parse-or-invalid) with no candidate naming anywhere.
- STABILIZER SWEEP UNBLOCKED (owner gated it on "the eventual menu-free
  recipe"; that recipe is now a documented negative result, so the sweep
  runs on the standard with-menu frozen recipe). Stage 1 screening
  launched on gruenau1 (round `2026-08-08-intent-stabilizer-sweep-v1`,
  snapshot `6cf9424`, joint-training screening — frozen-protocol
  replication for survivors): stab-online-nosg (collusion control, no
  EMA/stopgrad targets), stab-ldad-only (Delta-JEPA claim: LDAD, no
  EMA/stopgrad, no VICReg), stab-sigreg (SIGReg replaces VICReg).
  Reference rows already exist (recipe 5 seeds = EMA+stopgrad+VICReg).
  SIGReg objective existed in-tree; VISReg port still pending (stage 2).
  LDAD + full recipe (EMA+stopgrad+VICReg+LDAD, 1 seed):
  .213/.757/.867/.943/.960 — a LARGE shallow-depth gain (D2 .757 vs
  recipe .450, D1 .213 vs .126), deep on par with SIGReg. Matches theory
  T4 (best-of-R label weakest at H=1): LDAD's action-displacement
  decoding plausibly sharpens one-step geometry. REPLICATED at 3 seeds:
  D2 .741+-.021 (vs recipe .450+-.031), D16 .972+-.012 (vs SIGReg
  .959+-.011) — LDAD+recipe is the best variant overall so far.
  SIGReg+LDAD (i.e. LDAD with VICReg REPLACED by SIGReg, 1 seed): does
  NOT compose — .130/.403/.783/.840/.843, worse than either alone.
  VICReg seems necessary for the LDAD gain. Candidate headline:
  LDAD+EMA+stopgrad+VICReg; seeds 3-4 + frozen-protocol distill from the
  LDAD backbone running on gruenau1.
  STAGE-1 RESULTS (strict D1..D16, 1 seed): online_nosg+VICReg
  .093/.317/.670/.727/.727 and LDAD-only .080/.223/.637/.723/.720 — both
  train WITHOUT collapse (Delta-JEPA's no-EMA/no-stopgrad claim holds for
  stability) but lose ~.16 deep vs the reference .884: EMA+stopgrad buys
  real performance here, not just anti-collapse. SIGReg (replacing
  VICReg, EMA+stopgrad kept): 3 SEEDS .877/.917/.873 at D4 and D16
  .950/.970/.950 — D16 .957+-.012 vs reference .884+-.023: a REPLICATED
  ~.07 deep-depth improvement over the frozen recipe. Seeds 3-4 running
  to reach the 5-seed headline convention; if they hold, the paper
  recipe becomes SIGReg (candidate `mix4_aux025_nohorizon` +
  objective.sigreg.weight=1, vicreg 0) and downstream tables re-anchor.
- Learned action prior IMPLEMENTED (commit bb300f5): GaussianActionPrior
  head (state -> Gaussian over action embeddings, NLL on observed actions),
  planner `candidate_interface=learned_catalogue` (prior-filtered catalogue
  top-k/top-p, no menu, no feasibility oracle), `mean_prior_rank`
  diagnostic; suite 731 green. First menu-free training cell
  `mix4-prior-s0-v1` (variant `mix4_aux025_nohorizon_prior`, snapshot
  `f27403a`) RUNNING on gruenau1 GPU0: recipe + full-catalogue ranking
  candidates + prior + support head, eval fully menu-free (top-k 8).
- Five-seed headline slack-curve re-eval COMPLETE
  (`2026-08-07-intent-headline-slackcurve-v1/aggregate_slack_curves.json`):
  strict means reproduce the frozen headline exactly
  (.126/.450/.837/.879/.884); mean excess steps when solved 2.04/0.96/
  0.24/0.17/0.15 at D1..D16. Figure-ready.
- Frozen-backbone triangle now 2-SEED on all three legs (s1: plain
  .107/.250/.643/.797/.803; TD-shaped .143/.383/.893/.917/.910;
  ranking-shaped .133/.390/.807/.830/.830 — all replicate s0).
- `run_intent_terminal_energy_eval.sh` now does ONE generous run per depth
  (MAX_SLACK=4, `slack_curve=true`): exact success at every slack 0..4 +
  per-episode excess steps (`slack_curves` in metrics.json; legacy per-slack
  rows kept for collectors). ~2x cheaper eval; use everywhere.

## Next steps (priority order)

1. Seed replication for paper tables: faithful baselines (td_jepa,
   goal_head), frozen-backbone (both inits), K sweep — 3-5 seeds each as
   GPUs free up. All evals via the slack-curve script.
2. Faithful iGSM + near/far length OOD as the primary generalization axis;
   looped-LM test-time-compute comparison with real FLOPs.
3. Theory T3 (margins vs compounded drift): needs light planner
   instrumentation to record root-candidate score gaps per depth; then the
   saturation-depth prediction test. T1/T2/T4 done (see results above).
4. Re-run five-seed headline evals with the slack-curve evaluator for full
   excess-step curves in the paper figures.
5. Stabilizer sweep (owner 2026-08-07, run AFTER menu-free recipe locked):
   backbone variants {EMA/stopgrad/online_nosg} x {VICReg/SIGReg/VISReg/
   LDAD/none} under the FROZEN distillation protocol (removes the EMA-label
   confound; labels from the frozen encoder). Stage 1: ~8 single-seed
   screening cells incl. Delta-JEPA claim (LDAD, no EMA, no stopgrad) and
   collusion control (online_nosg). Ports: SIGReg/VISReg/latent-LDAD from
   sibling tracks; state_target switch already in DiscourseJEPA.
6. Causal-predictor endpoint-Energy cell (needs training-time recursive
   imagination for the transformer predictor; guard currently blocks it).
7. Transfer to a second established environment (multi-hop logical
   reasoning) once the iGSM matrix is locked.

## 2026-08-11: open-ended proposers built; domains unlocked; length-OOD running

- Owner directives: (a) implement all four contract domains (faithful iGSM,
  ProofWriter, PlanBench Blocksworld, text ALFWorld) via supervised
  subagents; (b) screen MULTIPLE fully open-ended proposal mechanisms
  (generator head AND CEM-in-embedding-space with decoder grounding +
  off-manifold control); (c) launch every paper-ready experiment in
  parallel. Confirmed: nothing had been trained on faithful iGSM yet.
- Open-ended proposers implemented + merged on main (5f89e47..f0fae8c):
  generator_cycle (state-conditioned AR phrase head, detached, sampled
  proposals vetted by LDAD cycle) and cem_cycle (CEM over action embeddings,
  training-embedding Gaussian prior pinned pre-eval-override, scale-invariant
  off-manifold penalty, LDAD decode for grounding). One shared decode/score
  module (planning/ldad_decode.py), one phrase parser (igsm/render.py),
  unified proposal diagnostics (proposal_recall, proposal_parse_rate —
  NOTE: parse-rate denominator differs generator=all samples vs CEM=elites).
  803 tests green. Adversarial review agent with failing-test mandate
  running over the full diff before any GPU spend.
- Observed-action planner merged: plan.py can now run slack-curve evals for
  ProofWriter/Blocksworld/ALFWorld through the same endpoint-Energy code
  path as stylized iGSM (evaluate_observed_action_planning; oracle row =
  privileged expert replay, labeled). Faithful iGSM evaluator gained
  single-pass slack_curve. Slack-curve outputs now get _slackcurve suffix.
- ProofWriter real-scale data compiled: data/intent_phrase/proofwriter/
  19884/1962/2000 episodes, depths 1/2/3/5 (OWA, ProofWriter's own splits),
  validation clean (replay/goal/catalogue-recall 1.0, disjoint splits),
  smoke-trains. MANIFEST.json has provenance. Blocksworld needs more
  instance sets + split discipline; ALFWorld needs large collection run.
- Length-OOD (runs/autonomy/intent_phrase/2026-08-11-intent-length-ood-v3):
  LDAD seeds s0-s4, feasible_menu lookahead 16. Train lengths 3-9; bands
  near 10-12 (nvars 20-28), far 13-16 (32-48), extreme 17-20 (40-64),
  strict length sampling. Two earlier launches were WRONG (EVAL_DEPTHS is
  lookahead, not problem length; default graphs can't reach length 10+) —
  discard 2026-08-11-...-v1. First result s1 near: mean_necessary 10.65,
  strict .047, slack-4 .233 (vs ~.96 strict in-dist) — big length drop.
  Graph-size confound control queued: length 3-9 at nvars 20-28
  (ctrl-nvars cells, auto-start when band evals finish).
- Ops notes: this session runs ON gruenau1 — pkill -f patterns match our own
  monitors/shell (use PID kills or bracket-escaped patterns). Parallel
  "predictive-state" subproject commits on main are NOT intent_phrase; leave
  alone.

## 2026-08-11 (cont.): probe verdict; codebook stage-1 negative; decoder stop-token fix

- STATE-READOUT CONTROLS (research/reports/intent_phrase/
  2026-08-11-state-readout-controls/): (A) "info absent from state" REFUTED
  — feasibility probe AUC .935, resolvedness .982 (oracle labels,
  candidate-privileged diagnostic). Residual is the OBJECTIVE: four
  imitation-trained heads (71k..1.65M params, incl. cross-attention over the
  sentence sequence) all stall >=0.48 nats above the feasible-entropy floor
  with feas. AUC <=.83, while a directly supervised same-shape MLP hits
  .935. One-hot ablation proves the conjunctive parent-lookup is the hard
  part (.935 -> .755). Campaign-log + interface-report overclaims corrected.
- CODEBOOK (owner idea, stage 1 eval-time k-means over training action
  embeddings, candidate_interface=codebook_cycle, committed da421db):
  NEGATIVE on existing LDAD checkpoints. Two causes found: (1) LDAD decoder
  free-decoding never stops (PAD position unsupervised — same defect family
  the adversarial review fixed in the generator loss); mitigated eval-time
  by terminator ('.') truncation in greedy_phrases. (2) After the fix,
  grounding is the blocker: codes decode to plausible but WRONG-problem
  variable names (1/64 parseable at a real state; own-catalogue embeddings
  reconstruct exactly teacher-forced but only 2/9 free-decoded). Free
  decoding from off-catalogue embeddings is not grounded by state
  conditioning on current checkpoints. Stage 2 options if pursued: supervise
  LDAD PAD position (retrain), or rely on the generator head (trained for
  free generation, PAD fix in). Screening evals eval-codebook-k{64,128}
  recorded the honest 0-parse numbers pre-fix.
- Length-OOD first number (s1, near band 10-12 steps, nvars 20-28):
  strict .047, slack-4 .233 (in-dist ~.96 strict) — large drop; graph-size
  control cells queued.

## 2026-08-11 (cont. 2): frozen-state sentence decoder (owner-requested renderer)

- Report: research/reports/intent_phrase/2026-08-11-state-decoder/ (mirrored).
  JEPA-pure readout (frozen backbone, zero-grad asserted). TRUE states:
  77% exact sentence / 96% token / 96% VALUE accuracy; final-answer head 94%
  (chance 4%) — states are a faithful decodable record incl. arithmetic.
  IMAGINED states: form survives (token .92 at d1-2), values collapse
  (.32 -> chance by d5). Controls: prompt-only floor .248 value (one
  predictor step buys +7pts); one-step-from-true-prefix NO better than
  compounded rollout at matched depth => SINGLE-STEP fidelity, not
  compounded drift, is the bottleneck; predictor contributes structure
  (token .561 -> .923), not computation. Paper story: predictor propagates
  plan-relevant geometry (what cycle-consistency/Energy read), not
  arithmetic; explains why calibrated-value baselines fail with depth.
  Feeds theory T3 (drift is immediate, not accumulated).

## 2026-08-11 (cont. 3): open-ended screen — generator negative, CEM eval launched

- gen-ldad-s0-v1 (LDAD recipe + state-conditioned action-generator head,
  PAD fix in, snapshot 318d6c2) trained clean (LDAD token acc .999,
  generator CE plateau 0.62/token) but generator_cycle planning fails:
  parse rate .132, no-usable-proposal in 98.7% of episodes, strict .013
  (random-with-menu .050). Direct sampling from the checkpoint shows the
  SAME grounding failure as the codebook: phrases are fluent, correctly
  formatted, but recombine adjectives/nouns across problems ("square
  beads", "large apples" for a problem that has neither) and are all leaf
  lookups; lowering temperature only shrinks the sample set, parseable
  fraction stays ~1-2 of 16. Consistent conclusion across codebook +
  generator: emitting THIS problem's variable names from the pooled state
  is the shared bottleneck for every catalogue-free proposer — same
  conjunctive-binding gap the state-readout probes localized. ldad_cycle
  (catalogue enumeration + cycle feasibility) remains the menu-free result.
- cem_cycle eval launched eval-cem-s0-v1 (frozen stab-ldad-ema-s0, fresh
  snapshot 734bc58 WITH the greedy_phrases terminator fix; pop 64, elites 8,
  3 iters, prior anchor 0.1). Note: first launch used a pre-fix snapshot
  and was killed before producing output. Expectation: same grounding
  ceiling (CEM decodes phrases from optimized embeddings), screen closes
  the "catalogue-free" question either way.
- cem_cycle result: parse rate .032, zero usable proposals in 100% of
  episodes, strict .000 — same grounding failure, worst of the three.
  SCREEN CLOSED: all catalogue-free proposers (generator/CEM/codebook) fail
  on emitting this-problem variable names; ldad_cycle (catalogue + cycle
  feasibility) is the paper's menu-free result. Report:
  research/reports/intent_phrase/2026-08-11-openended-screen/ (mirrored).

## 2026-08-11 (cont. 4): faithful-iGSM admission gate — one real defect found+fixed

- Ran the paper admission gate (run_igsm_paper_admission_gate.sh, tiny cells,
  gate-s0-v1). Passed: oracle 1.0 / random .11 strict bounds sensible,
  geometry beats random, dropout zero, EMA eval-mode invariants OK.
  DEFECT: the shuffled-actions control was bit-identical to the aligned cell
  — the faithful adapter (src/textjepa/data/faithful.py) silently dropped
  data.shuffle_actions via **_ kwargs. Fixed (80cb55e): alignment permuted
  last after all other randomness, stylized-dataset contract; regression
  test added (shuffle must change alignment, nothing else). Shuffled gate
  cell rerunning from the fixed snapshot; LR screen admitted once
  shuffle-hurts is confirmed.
- Shuffle-flag bug had a SECOND half: build_dataset's igsm_real branch
  enumerates kwargs and omitted shuffle_actions (fixed ce2581c, end-to-end
  verified). Tiny gate cell (64 problems) now differs but is ambiguous;
  scaled pair (512 problems, 200 eval episodes) running to settle
  shuffle-hurts. AUDIT NOTE for later faithful work: the faithful adapter
  also silently drops (via **_) stylized-only flags incl.
  invalid_action_mode, steps_range/strict_steps_range,
  geo_rank_factual_only, geo_rank_candidate_interface — any faithful
  length-OOD or invalid-mode experiment must add explicit support first,
  not assume the config key works.
- Scaled shuffle pair (512 problems): BOTH arms at the random baseline —
  the tiny gate model does not learn planning at that scale, so the
  shuffle-hurts criterion cannot be shown in tiny cells. Resolution: test
  the falsifier at real recipe scale (contract requires it as a causal
  falsifier anyway). LAUNCHED faithful-iGSM compact LR screen
  (2026-08-11-intent-faithful-screen-v1): LDAD headline recipe on
  data=igsm_real, LRs {1e-4,3e-4,1e-3,3e-3} seed 0, PLUS
  shuffled-actions arm at 3e-4 (real-scale gate check). Sequential chain
  on gruenau1 GPU 0, order 3e-4 -> 1e-3 -> shuffle -> 1e-4 -> 3e-3;
  snapshot ce2581c (both shuffle fixes in). All other gate criteria
  passed (oracle/random bounds, dropout, EMA invariants, geometry>random
  in the overfit cell).
- Screen launch #1 failed at hydra parse: the shared cell script passes
  stylized-only data keys. Ported to the faithful adapter (42fd571):
  geo_rank_horizons multi-horizon sampling (stylized-identical RNG stream)
  + geo_rank_rollout_for_h1 implemented; candidate_interface/factual_only/
  feasible_k/invalid_k/invalid_action_mode accepted at defaults, raise
  NotImplementedError otherwise (no more silent drops). CPU dry-run of the
  full override set trains. Chain relaunched from snapshot 42fd571.

## 2026-08-11 (cont. 5): standup-figure baselines — LM under all three menu conditions

Owner wants the depth-scaling figure's baseline to be a real sentence/token
LM (not TD-JEPA/GoalHead) across all three interface conditions (feasible
menu, full-catalogue non-oracle, no-menu free generation), plus our current
LDAD headline checkpoint re-evaluated non-oracle for a fair 3-way comparison
figure per condition. Stylized igsm only (LM checkpoints `runs/lm_intent`,
`runs/sentlm_intent` are stylized; faithful full-catalogue LM path is out of
scope here, raises NotImplementedError).

Already on record before this round: token-LM/sentlm-LM feasible-menu
(slack0 strict): token .830, sentence .635 (`runs/lm_intent/
plan_slack0_lm_intent.json`, `runs/sentlm_intent/
plan_slack0_sentlm_intent_decoder.json`). No prior full-catalogue or
free-generation strict-success numbers existed for either LM baseline
despite the code paths existing (`plan_lm.py`/`plan_sentlm.py` always used
`env.feasible_actions()`; `eval_generative_lm_baseline.py` never had output
saved for these checkpoints).

Added `candidate_interface=full_catalogue` to `plan_lm.py`/`plan_sentlm.py`
(previously feasible-menu only): candidates become
`range(len(problem.vars))`, invalid picks go through
`env.step_or_invalid` (invalid=noop), `invalid_rate` added to output,
output filename gets a `_full_catalogue` suffix so it doesn't clobber the
existing feasible-menu json. Scoped to stylized igsm + score/target_kind
`intent` only (faithful + outcome-scoring raise). 703-suite-adjacent tests
(`test_fixed_budget_runner.py`, `test_oracle_cem_contract.py`) still pass;
smoke-tested both scripts at n_episodes=8 on both interfaces before the
real run.

Launched (round `2026-08-11-standup-baselines-v1`, from the working tree,
not a snapshot — eval-only, no training):
- `ours-fullcat-s0-v1` (gruenau, cuda:2): `stab-ldad-ema-s0-v1/model/
  best.pt` (the 5-seed headline checkpoint's seed 0), candidate_interface=
  full_catalogue, EVAL_DEPTHS="1 2 4 8 16", 300 episodes, beam width 8,
  via `run_intent_terminal_energy_eval.sh` (same script/protocol as the
  headline depth curve, so directly comparable).
- `lm-baselines/` (gruenau, cuda:0), sequential: token-LM and sentence-LM
  full_catalogue (slack0, 300 episodes) + token-LM and sentence-LM no-menu
  free generation (`eval_generative_lm_baseline.py`, 300 examples,
  width=8, `problem_solve_rate` is the strict-success field to read).

Note for whoever reads this next: the LM baselines have NO planning-depth
axis (greedy one-step-at-a-time policy, no imagined lookahead), so the
"3 figures" (one per menu condition) should show our depth curve plus a
FLAT reference line per LM baseline, not 3 comparable curves — don't force
a depth axis onto the LM numbers.
- Screen attempt #2 surfaced three more faithful-port defects (all fixed,
  deb1e5f): (1) CRITICAL - horizon-mode GAR requires ga_rollout_actions,
  which faithful rollouts never emitted, so geo_horizon_rank (the recipe's
  key ranking loss) silently skipped on igsm_real; the timed-out lr3e4
  cell trained with the loss at 0.0 throughout. Now emitted + alignment-
  tested; dry run shows the loss active. (2) FaithfulPlanner lookahead>1
  crashed on _sequences' absorbing None padding (action_text(None));
  lr1e3 trained fine and died in eval. (3) 5h/epoch at num_workers=2
  (faithful generation is CPU-bound): cell script now takes NUM_WORKERS;
  chain relaunched with 16 workers, all cells reset (earlier partial
  checkpoints trained WITHOUT the ranking loss - discarded).

## 2026-08-12: codebook deep diagnosis (owner question) — impossible in principle

- Report: research/reports/intent_phrase/2026-08-12-codebook-diagnosis/
  (mirrored). Phrase space is combinatorial: leaf lookups saturate (100%
  verbatim coverage), compute phrases NEVER repeat (0/1367 across 300 val
  problems; 0 problems with necessary actions covered) — so no codebook
  size can supply them. Twist: the LATENTS are nearly reusable (NN dist
  0.77 vs within-problem separation 10.7) — owner's intuition right at the
  vector level — but the environment executes sentences, and the retrieved
  vector's phrase names the wrong problem's variables. Action embeddings
  are coarse role/op codes; names live in the state, not the action code.
  Menu-free via prompt-derived catalogue + cycle feasibility stands as the
  right claim; retrieval memories are ruled out in principle.

## 2026-08-12: LM-baseline + scaling program stood up (PREPARED, nothing launched)

Audit of the contract's six learned LM rows: all six have a TRAINING path
already (`scripts/train_lm.py` with `train.target_kind=intent`,
`scripts/train_sentlm.py` with `model.latent_target`, and `model.recurrent`
for the three weight-shared counterparts). Planning eval under the
feasible-menu interface also existed (`plan_lm.py`, `plan_sentlm.py`), but it
did NOT produce the JEPA slack-curve metrics: it emitted a single fixed-slack
success and its own ad-hoc dict. Fixed (d52bae5): both now aggregate through
`textjepa.planning.evaluate.aggregate_episodes` with `slack_curve=true` and
record `solved_at` (exact for a greedy policy that stops at the goal), so
`success_by_slack` / `excess_steps` / `invalid_action_rate` /
`mean_necessary` match `scripts/plan.py` field-for-field. `invalid_rate` is
kept as an alias; curve runs get a `_slackcurve` filename suffix.
`configs/plan.yaml` gained `score` and `eval_loops`.

New `scripts/run_intent_lm_screen_cell.sh` is the LM analogue of
`run_intent_horizon_energy_cell.sh`: trains one row, then evaluates it under
the identical feasible-menu protocol, writing `metrics.json` with
`slack_curves`. It keys curves by RECURRENT LOOP COUNT (contract loops
{1,2,4,8,16}) and sets `planning_depth_axis: false`, so a loop count can
never be misread as a JEPA simulated-transition depth.

Trained LM checkpoints that already exist (usable NOW for the frozen-feature
state-readout probes — `scripts/export_intent_representations.py` already
supports `--kind token_lm|sentence_lm`): token LM d_model=288 and sentence LM
(+/- latent) d_model=256, seeds {1,2,3}, on BOTH stylized (`runs/lm_intent*`,
`runs/sentlm_intent*`, `runs/sentlm_latent_intent*`) and faithful
(`*_faithful*`) iGSM, all at lr 3e-4 (faithful token LM at 1e-3), 20 epochs,
train_size 100000. NOTE these are off-contract widths (288/256, not
128/256/512) and off-contract seeds (1..3, not 0..4), so they are probe/
diagnostic material, not headline rows. NO recurrent LM checkpoint exists.

PREPARED (not launched):
- `runs/autonomy/intent_phrase/2026-08-12-intent-lm-screen-v1/` — compact LR
  screen {1e-4,3e-4,1e-3,3e-3}, width 256, seed 0, stylized iGSM, for token
  LM and sentence LM (8 cells + `run_chain_gpu0.sh`). EPOCHS=10
  TRAIN_SIZE=30000 BATCH_SIZE=32, matching the JEPA screen cells.
- `runs/autonomy/intent_phrase/2026-08-12-intent-width-scaling-v1/` — JEPA
  LDAD recipe (`mix4_aux025_nohorizon` + observed_action_ldad) at widths
  {128, 512}, lr 3e-4, seed 0, stylized iGSM (2 cells +
  `run_chain_gpu1.sh`). Width key is `model.d_model`; verified to
  instantiate at 6.05M / 22.82M / 89.41M params for 128/256/512. Width 256
  is already covered by `stab-ldad-ema-s0-v1`.
Both rounds run from the immutable snapshot `runs/autonomy/_code/d52bae5...`.

Still unimplemented (documented, not fixed): the three recurrent LM rows have
no screen cells yet (the runner supports them via `token_lm_rec` /
`sentence_lm_rec` / `sentence_lm_latent_rec`, but no checkpoint has ever
been trained, so their loop axis is untested); full_catalogue LM baselines
still raise on faithful iGSM; the LM rows have no matched-FLOPs axis
(they have no lookahead to trade compute against).

## 2026-08-12: predictor-variant falsifiers (code fix + PREPARED cells)

The paper's causal-falsifier list needs two predictor variants of the LDAD
headline recipe on stylized iGSM: a causal-sequence predictor (transformer
world model over the whole state/action history) and a non-residual MLP
predictor (predicts the next latent directly instead of a delta).

Fix: horizon-mode GAR (the recipe's ranking loss, which imagines a rollout
by stepping the predictor over encoded action tokens) used to raise
"horizon GAR currently requires the matched MLP predictor". Root cause: it
called the predictor once per horizon step with a single (state, action)
pair. For the causal predictor that call is a length-one sequence — it
resets the position embedding and throws away the factual prefix
s_0..s_t / a_0..a_{t-1}, i.e. silently degrades it to a Markov MLP.
`DiscourseJEPA._horizon_prefix_endpoints` now rebuilds the factual prefix
per anchor step (grouping rollout rows by their anchor t) and uses the
predictor's own history-carrying `rollout`; masked post-terminal suffix
actions stay absorbing exactly as before. The MLP path keeps the literal
old recursion, pinned by
`test_horizon_gar_mlp_path_matches_explicit_markov_recursion` (compares
model output against the explicit `s <- F(s,a)` loop, so existing
checkpoints are unaffected). Two new causal tests check gradients flow and
that the endpoints equal the history-carrying rollout and differ from the
old history-free recursion. tests/test_model.py: 67 passed.
Rollout-exposed GAR (`geo_rank_rollout_depths`) still raises for the causal
predictor — untouched, unused by this recipe.

Smoke-tested both variants for 1 epoch on CPU (tiny sizes) through
`scripts/run_intent_horizon_energy_cell.sh mix4_aux025_nohorizon` with
observed_action_ldad: non-residual (`model.predictor_residual=false`) trains
and evaluates end to end; causal (`model.predictor_kind=causal`) trains with
`train/geo_horizon_rank` active (nonzero), i.e. the loss is no longer
skipped or blocked.

PREPARED (not launched):
- `runs/autonomy/intent_phrase/2026-08-12-intent-predictor-variants-v1/`
  `pred-causal-s0-v1` (model.predictor_kind=causal) and `pred-nonres-s0-v1`
  (model.predictor_residual=false), seed 0, lr 3e-4, EPOCHS=10
  TRAIN_SIZE=30000 BATCH_SIZE=32, stylized iGSM, N_EPISODES=300
  BEAM_WIDTH=8 EVAL_DEPTHS="1 2 4 8 16" MAX_SLACK=4, CUDA_VISIBLE_DEVICES=0,
  to be run on gruenau2 from an immutable snapshot under
  `runs/autonomy/_code/`.

## 2026-08-12 (cont.): LM state-readout probes — feasibility signal NOT JEPA-specific

- Report: research/reports/intent_phrase/2026-08-12-lm-state-readout/
  (mirrored). Token LM .971 / sentence LM .961 / sentLM+latent .947 vs JEPA
  .935 feasibility AUC (own action embeddings; matched 16-d candidate:
  sentence LMs land AT JEPA, token LM above). Resolvedness tied everywhere.
  Conjunctive parent-lookup collapse replicates on all families. LMs decay
  LESS with depth than the JEPA pooled state. CONSEQUENCE: drop any
  "JEPA states carry structure LMs lack" phrasing; representation story is
  about USE (cycle-consistency converts signal into planning; LMs have no
  analogous mechanism), presence is universal. Also honest JEPA weakness:
  deep-prerequisite feasibility decays in pooled state. Caveats: single
  seed, training-campaign confound, not the matched-width protocol.
- Also this session: codebook_ground interface committed (9f9f1b7), 300-ep
  eval running; predictor variants fixed+tested (fa63cc9: causal predictor
  was silently Markov inside horizon GAR — real bug), cells queued on
  gruenau2 waiter; LM LR screen (8 cells) + width scaling {128,512} chains
  launched on gruenau2 (metric-parity fixes d52bae5).

## 2026-08-12 (cont.): long-trace retrain (owner-ordered redesign of length OOD)

- Owner: train on 15-25-step traces (like official iGSM's longer op counts),
  eval bands 25-30 ... 45-50; cancel superseded runs. CANCELLED: v3 extreme
  reruns (s2/s3/s4), ctrl-nvars cells/waiters, LM OOD evals on 3-9-trained
  checkpoints. v3 near/far 5-seed numbers stay recorded (near strict
  .055±.062, slack-4 .282±.255; far ~0; extreme 0 on 2 seeds; protocol was
  feasible_menu + candidate-privileged future menus at lookahead 16).
- Design probing: stylized generator cannot produce >~37-step cones at
  default leaf_prob .35; leaf_prob=0.1 (held FIXED across train and eval so
  length is the only axis) makes all bands samplable fast: train [15,25]
  @ nvars [30,60]; eval [25,30]@[60,120], [30,35]@[90,160], [35,40]@
  [110,180], [40,45]@[130,200], [45,50]@[150,220]. Longest episodes ~205
  chunks -> model.max_chunks=256; batch 16.
- LAUNCHED 2026-08-12-intent-long-traces-v1 on gruenau1: jepa-ldad-long-s0
  (GPU2, LDAD recipe lr 3e-4), tok-lm-long-s0 (lr 3e-3) -> sent-lm-long-s0
  (lr 3e-4) chained on GPU1. Band evals to follow after training.
- codebook_ground 300-ep eval COMPLETED — results to be read/reported.

## 2026-08-12 (cont. 2): results batch + REVISED OVERALL PLAN (owner-confirmed)

Results landed:
- pred-causal-s0 (causal-sequence predictor, guard fixed): strict by depth
  1/2/4/8/16 = .327/.670/.850/.963/.970 — comparable to the MLP headline
  (five-seed .202/.729/.885/.965/.975), single seed. Causal predictor is
  NOT a failure mode; report as architecture variant. pred-nonres running.
- Width scaling (stylized 3-9, lr 3e-4 s0): w128 .933 / w256 .975 (headline)
  / w512 .990 strict at depth 16 — monotone width scaling.
- LM LR selection RESOLVED (2 seeds): token LM -> 3e-3, sentence LM -> 3e-4
  (mean success tie-breaks per contract). Long-trace cells already use these.
- codebook_ground 300-ep: slack-4 .26-.28, proposal recall 1.0 — matches
  ldad_cycle (.29); owner idea works, added to matrix below.

REVISED PLAN (stylized iGSM long-trace = primary mechanism environment):
- S1 screens (RUNNING): jepa-ldad-long / tok-lm-long / sent-lm-long at
  steps 15-25, leaf_prob 0.1, nvars 30-60; + JEPA LR cross-check when free.
- S2 mains: 5 seeds LDAD; TD-JEPA + GoalHead baselines; token/sentence/
  sentence+latent LMs AND looped (recurrent) counterparts (loops axis
  {1,2,4,8,16}, never depth curves). Slurm (Lise/Alex) packaging.
- S3 eval matrix: {ID 15-25, 25-30, 30-35, 35-40, 40-45, 45-50} x
  {feasible_menu, full_catalogue (no-oracle stress), ldad_cycle,
  codebook_ground} x depth/loops {1,2,4,8,16}. codebook_ground carried as a
  co-equal menu-free interface (evidence label: environment-side grounding).
- S4 ablations (contract list) incl. predictor variants (causal, non-res)
  on long traces; width {128,512} rerun at long traces for the main table.
- S5 DETACHED DECODER on everything: frozen-state sentence decoder
  (JEPA-pure readout, zero-grad, as 2026-08-11) trained per final main
  checkpoint (and per domain), so every reported system renders executed
  plans to text; report decode accuracy alongside planning numbers. LM
  rows already produce text natively — matched decoder capacity per the
  contract's frozen-feature decoder protocol.
- Domains (faithful screen running; ProofWriter/PlanBench/ALFWorld agent
  finishing gates) and probe battery on final five-seed checkpoints
  unchanged from before.

## 2026-08-12 (cont. 3): plan corrections (owner)

- S5 CORRECTED: detached decoder is NOT a per-checkpoint reporting layer —
  it is the execution engine of the fully autonomous no-menu mode: propose
  (cycle/codebook) -> decoder EMITS next-step text -> model re-encodes its
  own text -> repeat until claimed done -> check FINAL ANSWER only. No
  oracle executor inside the episode. One decoder per main checkpoint
  (JEPA-pure, frozen states). NEW BUILD ITEM: autonomous self-rollout eval
  mode (planner loop over own generated text + answer check). Preregistered
  expectation from decoder controls: structure survives, arithmetic is the
  weak link -> honest end-to-end number whatever it is. Completes the
  interface ladder: feasible menu -> no-oracle menu -> menu-free w/ oracle
  executor -> fully autonomous text generation.
- Interface axis clarified: ONE training run per seed serves all
  interfaces; codebook/prior are eval-time fits on the frozen checkpoint
  (never trained in). S3 stands as 6 bands x {feasible_menu,
  full_catalogue, ldad_cycle, codebook_ground, autonomous} where applicable.
- Causal predictor: on-par result (.970 vs .975 strict@16) -> ablation
  table entry (recipe not predictor-specific), no headline treatment.

## 2026-08-12 (cont. 4): S2 mains + ProofWriter screen SUBMITTED to Alex

Local Gruenau GPUs are saturated, so the paper-ready training campaigns were
packaged for Slurm. Alex only: every Lise job pends with
`AccountOutOfComputeTime` (account bem00089 budget exhausted, confirmed live),
so Lise is unusable for overflow until compute time is restored.

Alex deployment (own filesystem): immutable snapshots under
`/home/atuin/c107fa/c107fa12/TextJEPA-autonomy/_code/<sha>` pushed with
`scripts/cluster/sync_snapshot_to_alex.sh`; runs under
`.../TextJEPA-autonomy/runs/autonomy/intent_phrase/<round>/<cell>/` with the
same `job.sh`/`state`/`exit_code`/`stdout.log`/`stderr.log`/`environment.json`
layout as Gruenau, so `scripts/cluster/sync_back_round.sh <round>` drops
results straight into the local tree. Shared env
`/home/atuin/c107fa/c107fa12/.venv` (torch 2.9+cu128) verified by running the
LDAD cell AND both untested recurrent LM rows end to end on CPU there first.
Cells queue with `--partition=a40,a100,rtxpro6k --gres=gpu:1`: a single-a40
submission was estimated to start six days out, across three partitions the
first six cells started within minutes. Alex caps single-node jobs at 24 h
(`--time=23:55:00`, inner watchdog 84600 s) — the 48 h wish for the recurrent
rows is not grantable there; if a rec cell TIMEOUTs it needs a resume path.

- A. `2026-08-12-intent-long-mains-v1` LAUNCHED, 45 cells, jobs 3989961-3990010
  (snapshot 5213602). Long-trace stylized iGSM (steps 15-25, nvars 30-60,
  leaf_prob 0.1, strict, max_chunks 256 / token LM max_len 4096), EPOCHS=10
  TRAIN_SIZE=30000 BATCH_SIZE=16 N_EPISODES=300 MAX_SLACK=4, JEPA depths
  {1,2,4,8,16}. Rows x seeds 0-4: LDAD recipe (lr 3e-4, one knob `LDAD_LR`
  pending the local 3e-4 vs 1e-3 cross-check), TD-JEPA and GoalHead baselines
  (lr 3e-4, as in the 2026-08-07 baseline round), token LM 3e-3, sentence LM
  3e-4, sentence+latent 3e-4, and the three RECURRENT LM rows (loop axis) with
  seed 0 as smoke and seeds 1-4 held on `--dependency=afterok`. No ablations.
- B. `2026-08-12-intent-proofwriter-lr-screen-v1` LAUNCHED, 12 cells, jobs
  3990030-3990041 (snapshot a2f035e); compiled ProofWriter data rsynced to
  Alex. LDAD + token LM + sentence LM at lr {1e-4,3e-4,1e-3,3e-3}, seed 0,
  EPOCHS=30. ENTRY-POINT FINDING (checked on CPU first, as ordered): the
  stylized-iGSM cell `run_intent_horizon_energy_cell.sh` CANNOT run
  ProofWriter — it hard-codes on-the-fly iGSM sampling keys absent from
  `configs/data/proofwriter.yaml` (`Key 'geo_rank_rollouts' is not in struct`,
  aborts before step 1) and its evaluator samples iGSM problems. Fixed by
  using the compiled observed-action path: added a `proofwriter` case to
  `run_compiled_domain_ldad_cell.sh` and wrote `run_compiled_domain_lm_cell.sh`
  (train_lm/train_sentlm with `data=proofwriter train.target_kind=intent`, then
  `eval_observed_action.py`, same information-matched interface as the JEPA
  row). All three cells smoke-tested end to end; 31 related tests pass.
  CAVEAT: compiled-domain evals report `metrics_by_excess_actions`, not the
  iGSM `slack_curves` shape; recurrent / sentence+latent LM rows are not wired
  for compiled domains.
- C. `2026-08-12-intent-faithful-mains-v1` PREPARED ONLY on Alex (5 seeds,
  `state=PREPARED`, lr placeholder `PENDING_LOCAL_SCREEN`); launch with
  `FAITHFUL_LR=<lr> SUBMIT=1 bash .../alex_submit_intent_faithful_mains.sh`
  once the local faithful LR screen decides.

New helpers (committed): `scripts/cluster/{alex_env.sh,sync_snapshot_to_alex.sh,
alex_submit_intent_long_mains.sh,alex_submit_proofwriter_lr_screen.sh,
alex_submit_intent_faithful_mains.sh,sync_back_round.sh,
write_cell_environment.sh}`. Per-round READMEs with the exact rsync-back
command live in each local round directory.

## 2026-08-12 (cont. 5): PlanBench Blocksworld + text-ALFWorld corpora built, admission gates run

Report: `research/reports/intent_phrase/2026-08-12-planbench-alfworld-unlock/`.
Round dir: `runs/autonomy/intent_phrase/2026-08-12-intent-domains-v1/`.

CORPORA (both frozen, with provenance MANIFESTs; `data/` is gitignored):
- PlanBench Blocksworld `data/intent_phrase/planbench_blocksworld/`: 3000 train
  / 60 val / 80 test episodes + a 48-episode 6-block length-OOD test set.
  Instance sets `generated_basic_3`, `generated_basic`, `generated` (official
  pool 263). Identity = (initial, goal) atom set minimised over all block
  RENAMINGS, so relabelled duplicates cannot straddle splits; val/test are
  official instances only. Rejections recorded (2420 held-out-identity, 505
  renaming-duplicate, 1683 literal-duplicate, 335 block-count).
- text-ALFWorld `data/intent_phrase/alfworld/paper_v1/`: 840 train / 80 val /
  80 test episodes, 12204/1111/1244 transitions, 48816/4444/4976
  counterfactuals (half deliberately infeasible). Splits are the official
  train / valid_seen / valid_unseen game sets, zero gamefile overlap, all six
  task families present in every split. 100% of counterfactual branches carry
  `teacher_rollout_actions`. Collected by 48+8+8 deterministic shard stripes;
  40 of 48 train stripes completed (2 wedged on games that outlive the 900 s
  episode timeout, 6 were still slow-collecting when the round was frozen), so
  the corpus is 840 rather than 1000 episodes. Re-running the missing stripe
  indices extends it reproducibly; contributing shards are listed per split in
  MANIFEST.json.

INFORMATION BOUNDARY (the thing reviewers will attack): ALFWorld's
`admissible_commands` and both domains' expert plans are COLLECTION labels
only. Deployment scores a separately generated grounded catalogue
(`observed-entities-v1` for ALFWorld: 35-2895 actions/step; formable block
operations for PlanBench: 18-50). Collection FAILS HARD if that catalogue
omits the expert action, pinned by
`tests/test_alfworld_adapter.py::test_collection_fails_when_grounded_catalogue_omits_expert_action`,
so 100% catalogue recall is an invariant, not a survivor average.

ADMISSION GATES — both domains pass every DATA-side gate, both FAIL the
tiny-overfit gate. PlanBench (`_gate_planbench_v2`, 6/8 checks):
schema/replay/recall/disjointness pass (exact replay 1.0, catalogue recall
1.0, goal success 1.0 on ALL 3000+60+80 episodes); horizon Energy live;
random 0.0 vs oracle 1.0; dropout zero + EMA eval pass; closed loop runs on
both interfaces. But a 0.36M model trained 400 epochs on 24 episodes reaches
strict success 0.0 on its own TRAINING set (random 0.0), while memorising the
data well (`observed_action_sequence_exact` 0.702, token acc 0.950,
`goal_dist_corr` 0.579) and emitting invalid actions 92% of the time on the
full catalogue. Length-2 control (`_gate_planbench_len2`, 7/8): with only
2-step plans and near-perfect geometry (`goal_dist_corr` 0.995) the model gets
0.167 vs RANDOM 0.250 — i.e. the failure is NOT plan-length compounding, it is
that the endpoint-Energy selection does not beat random on this domain at tiny
scale. The shuffle falsifier IS wired and bites hard at the loss level
(sequence-exact 0.702 aligned vs 0.013 shuffled); it only scores "false" in
the summary because that check compares strict success, which is at floor for
both cells. ALFWorld (`_gate_alfworld`): data gates pass (exact replay 1.0,
recall 1.0, goal success 1.0 on 40 sampled episodes/split — full live-engine
replay of 1000 episodes is hours), horizon Energy live at h=1 and h=8,
random 0.0 vs oracle 1.0; learning-side cells (40 epochs) were still running
when this entry was written — see `_gate_alfworld/gate_summary.json`.

CONSEQUENCE: per the admission rules in PAPER_EXPERIMENTS.md neither domain is
admitted for the LR sweep yet. Both first cells are PREPARED, NOT LAUNCHED:
`2026-08-12-intent-domains-v1/{planbench,alfworld}-ldad-lr3e4-s0-v1/job.sh`
(LDAD, lr 3e-4, seed 0, width 256, `state=PREPARED`, snapshot under
`runs/autonomy/_code/<sha>`). Open question for the owner: whether the
tiny-overfit criterion is the right admission bar for domains whose plans are
7-15 steps long, or whether it should be scored on a non-floored metric
(LDAD sequence-exact / goal_dist_corr), which both domains pass clearly.

Gate-tooling fixes committed this round: `TINY_DATA_ROOT` so the tiny cells can
use a cheap subset while replay/bounds still read the full corpus; and
`check_compiled_domain_horizon_loss.py` no longer hardcodes iGSM's
`max_chunk_len=96`, which had made the ALFWorld gate impossible to run at all.

## 2026-08-12 (cont. 4): monitoring sweep advances

- Predictor variants COMPLETE (report 2026-08-12-predictor-variants/,
  mirrored): non-residual strict@16 .980, causal .970 vs headline .975 —
  recipe is predictor-agnostic; ablation-table entry. Task #7 closed.
- Faithful screen interim: 3e-4 strict@16 .243/slack4 .707 beats 1e-3
  (.210/.563). NOTE: depth curve is FLAT on faithful (d1 .263 -> d16 .243)
  — planning depth buys nothing on faithful iGSM so far; hold
  interpretation until screen + 5 seeds. Shuffle arm running; 1e-4, 3e-3
  queued.
- sent-lm-long-s0 (15-25 steps) COMPLETED early (ran in first chain):
  loops1-decoder success recorded in its metrics.json; note current chain
  may rerun it after tok-lm — harmless overwrite, same config.
- Alex: 12 running / 49 pending (mains + ProofWriter screen).

## 2026-08-12 (cont. 5): long-trace first mains + faithful shuffle verdict

- Alex long-trace mains landing (synced back): at slack 4 EVERYTHING is
  near floor on 15-25-step problems — jepa-ldad-long s0 .067 slack-4 @d16
  (monotone in depth, ~10x baselines), other seeds .013-.03; tok-lm .010,
  sent-lm .007 (local screens agree); TD-JEPA and GoalHead .000-.007
  everywhere. Slack 4 on ~20 necessary steps is likely the wrong budget
  (contract metric is the full success-vs-budget curve); LAUNCHED
  wide-slack (16) eval of jepa-ldad-long-s0 on gruenau2 GPU1
  (eval-wideslack-jepa-s0-v1) to find where curves separate before
  touching training budgets.
- FAITHFUL HOLD: shuffle falsifier at real scale does NOT hurt strict
  (.247 shuffled vs .243 aligned; slack-4 .610 vs .707; depth curve flat
  both). Faithful 5-seed mains NOT triggered. Needs diagnosis: the ~.24
  faithful strict may be interface+heuristics, not grounded planning —
  echoes the compiled-domain finding (selection below random at tiny
  scale). tok-lm-long local COMPLETED: strict .000/slack4 .010 (matches
  Alex).
- Local jepa-ldad-long-s0 COMPLETED, agrees with Alex twin (slack4 .073
  @d16, strict 0; mean necessary 16.9). Launched eval-bands-jepa-s0-v1 on
  freed gruenau1 GPU2: all 6 bands (ID 15-25 through 45-50) at slack 16
  with full curves (subsumes any smaller budget); 200 eps/band. Wide-slack
  ID eval still running on gruenau2.
- Faithful REVERSAL: lr 1e-4 wins (.337 strict @d16) AND restores depth
  monotonicity (.247 d1 -> .337 d16), vs flat .24 at 3e-4. The negative
  shuffle verdict was measured at the WRONG LR; queued
  faith-ldad-shuffle-lr1e4-s0-v1 behind the last screen cell (3e-3).
  Faithful mains remain held until shuffle@1e-4 is in.
- Long-trace LR CONFIRMED 3e-4 (1e-3 crosscheck: slack4 .037 @d16 vs
  .073) — Alex mains already at 3e-4, no action.
- ALFWorld gate_summary.json: all data-side checks pass; tiny-overfit and
  shuffle fail on floored strict (as reported) — owner decision pending.
- ProofWriter LM screens (Alex, synced): all 8 LM cells strict .61-.65,
  slack4 .94-.97 — LMs are STRONG on ProofWriter; PW LDAD cells still
  running. The JEPA-vs-LM comparison on PW will be the interesting one.
- WIDE-SLACK VERDICT (eval-wideslack-jepa-s0): the long-trace regime is
  HEALTHY — jepa-ldad-long-s0 rises smoothly to .72 success at slack 16
  (~17 necessary steps) vs first-feasible .26 and random .13. Slack 4 was
  simply the wrong reading point for 20-step plans; headline metric
  becomes the full success-vs-budget curve + AUC (as the contract already
  specifies). No training-budget escalation needed. Band evals (slack-16
  curves per band) mid-run; ID band agrees (.69@16, 200 eps). LM
  wide-slack ID evals launched on CPU (tok/sent long checkpoints).
- Rec-row LM smokes passed on Alex; all rec seeds COMPLETED (sync+record
  next sweep).
- Long-trace WIDTH ladder submitted to Alex (2026-08-13-intent-long-width-v1):
  jepa-ldad-long at d_model 512 and 768, seed 0, same long-trace data —
  direct test of whether length extrapolation is capacity-bound (current
  mains are 8.6M params / width 256; LM baselines 7.2-7.4M, matched).
  Jobs 3998549/3998550.
- Wide-slack LM comparison landing: sentence LM reaches .26 at slack 16 on
  the ID band vs JEPA .72 (first-feasible .26, random .13) — the headline
  separation, on the curve metric.

## 2026-08-13: capacity ladder + domains unblocked (owner decisions)

- Owner: "screw the gate" — PlanBench + ALFWorld admitted despite the
  tiny-overfit-on-strict-success failure (bar is structurally unpassable
  for 7-15-step plans; non-floored metrics pass clearly, incl. shuffle
  falsifier .702 vs .013). LAUNCHED planbench-ldad-lr3e4-s0-v1 (gruenau1
  GPU0) and alfworld-ldad-lr3e4-s0-v1 (GPU1). Report the strict-success
  gate failure honestly in the paper.
- CAPACITY LADDER on long traces (Alex, 2026-08-13-intent-long-width-v1),
  all seed 0, same long-trace data:
  * width: d_model 512 (job 3998549), 768 (3998550)
  * depth: state_layers 8 (3998593), 12 (3998594), predictor_layers 4
  * GPT2-small-shaped: d_model 768 x 12 layers, batch 8 (3998595)
  Headline reference is d_model 256 / state_layers 4 / 8.6M params;
  LM baselines 7.2-7.4M (matched). Repo's own igsm-124m run used
  d_model 768 / 11 layers, i.e. GPT2-small shape = the iGSM-paper scale.
  Rationale: the 45-50-step band has 150-220 prompt sentences compressed
  into one 256-d state, so state width AND depth (multi-hop within the
  layer stack) are both plausible binding constraints; the ladder
  separates them.

## 2026-08-13 (cont.): BASE SIZE FIXED from the iGSM paper + faithful cleared

- Read Physics of LMs 2.1 (arXiv 2407.20311): their model is GPT2-small
  shape — 12 layers x 12 heads x 768 dim, 124M params (RoPE). KEY
  ARCHITECTURE FINDING: **depth matters more than width** — "a 4-layer
  transformer, even with 1920 hidden dims, underperforms", while
  "20-layer 576-dim performs very well"; layer-by-layer recursive
  dependency resolution (shallow layers -> near params, deep -> distant).
  OUR HEADLINE WAS 4 STATE LAYERS = exactly their failing config.
  Their OOD design: iGSM-med train <=15 ops -> test 20/21/23/24;
  iGSM-hard train <=21 -> test 28-32, i.e. OOD ~1.5x train max. OUR bands
  went to 2-3.3x train max — more aggressive than anything reported.
- DECISION (owner): adopt GPT2-small shape as BASE for everything
  (JEPA state_layers=12/d_model=768/heads=12/predictor_layers=4; token LM
  n_layers=12/d_model=768; sentence LM state_layers=12/d_model=768),
  batch 8. New round 2026-08-13-intent-base-mains-v1 (45 cells generated);
  seed-0 cells of all families submitted first as smoke, rest to follow.
  PlanBench/ALFWorld will use the same base size for their mains.
  Capacity ladder (width 512/768 vs depth 8/12) kept running as the
  width-vs-depth ABLATION replicating the paper's claim in our setting.
- FAITHFUL CLEARED: at the correct LR (1e-4) the shuffle falsifier DOES
  hurt — aligned .337 strict / .837 slack4 @d16 vs shuffled .277 / .690,
  and only the aligned model gains with depth (.247->.337 vs .233->.277).
  The earlier "shuffle doesn't hurt" verdict was an artifact of the wrong
  LR (3e-4). Faithful LR = 1e-4; mains unblocked (to be submitted at base
  size). 3e-3 also completed (.283 @d16) — 1e-4 is the winner.

## 2026-08-13 (cont. 2): moved off Alex; paper-exact bands; ProofWriter hardened

- ALEX FREED (owner needs it): all intent_phrase jobs cancelled (base
  mains, capacity ladder, PW screen); completed cells synced back with
  checkpoints. Compute moves to Grünau — gruenau7 has 4 free A6000 (49 GB)
  plus gruenau2 mostly free.
- PAPER-EXACT BANDS: probing showed our STYLIZED generator cannot express
  their spec — capped at their instance-parameter budget (n_vars<=28) it
  yields median 4-5 necessary steps, max ~14, and op 28-32 needs n_vars
  30-60. Their "op" counts solution operations, not resolved variables.
  The FAITHFUL adapter takes their actual knobs, and verified working:
  iGSM-hard train (max_op=21,max_edge=28,op_range[3,21]) and OOD
  op_range[28,32] both sample fine (3.4 s/problem at OOD, acceptable).
  DECISION: the paper-matched generalization experiment runs on FAITHFUL
  iGSM-hard; stylized stays the mechanism/dev environment with its own
  band design as a secondary axis.
- LAUNCHED 2026-08-13-igsm-hard-base-v1 on gruenau7: hard-ldad-lr1e4-s0
  and hard-ldad-lr3e4-s0 (GPT2-small shape: d768/12 layers/12 heads,
  predictor 4, batch 8, MAX_SLACK=16 so the curve metric is native).
- PROOFWRITER HARDENED (owner: saturation tells us nothing): built
  data/intent_phrase/proofwriter_depth_ood/ — train/val source_depth<=3
  (16711/1500), test = depth-5 ONLY (500), plus test_id depth<=3 (1500)
  for the in-distribution contrast; config
  configs/data/proofwriter_depth_ood.yaml. Depth generalization replaces
  the saturated mixed-depth benchmark as the headline PW setting.
- Non-residual predictor -> ablation table (with causal), per owner.

## 2026-08-13 (cont. 3): new length-controlled logic domain ported from synthetic-RLVL

- NEW DOMAIN `fsa-deduction` (data/intent_phrase/fsa_deduction/), ported
  from the owner's own `alex:~/synthetic-RLVL` generator
  (`synthetic_dataset.py`, `_generate_hard_fsa_core`, difficulty
  `hard_fsa`; plan doc `docs/hfsa_depth_scaling_plan_2026-05-19.md`).
  Each problem unrolls a finite-state automaton over `depth` layers: c0
  gets a state word and a marker word, every layer offers K=4 textually
  plausible branch rules but only one is derivable, and the gold
  derivation is 2*depth-1 forward-chaining steps. `depth` is the single
  length knob and is effectively unbounded. Replaces ProofWriter as the
  logic domain that can be pushed to 50 reasoning steps.
- Bands follow their convention (band_train = step <= train_max, band_ood
  = step > train_max): train/val/test_id at 15-26 layers (29-51 steps),
  OOD test bands 27-32, 33-40, 41-50 (up to 99 steps).
  Counts: train 2000 / val 300 / 200 per test band; 579 MB.
- Compiles to the existing observed-action schema and reuses the
  ProofWriter Datalog executor, so training/eval run unchanged. Unlike
  ProofWriter it DOES record `teacher_rollout_actions`, so the horizon
  ranking loss is live (checked: `horizon_ranking_active: true`,
  12 rollout action steps) — this is the loss the log flagged as
  silently zero on the older compiled domains.
- Data gate all green: schema OK, expert-in-catalogue 1.0, exact replay
  1.0, goal success 1.0, splits disjoint by content hash, deterministic
  regeneration verified. Reusable script
  `scripts/validate_fsa_deduction_corpus.sh`.
- CAVEAT to report honestly: by construction exactly one rule is
  applicable at each point, so the `feasible_menu` eval interface is
  degenerate on this domain; the meaningful interface is the full
  non-oracle catalogue (115-206 actions ID, 323-398 in the 41-50 band).
  `--side-facts-per-step` exists if a non-degenerate feasible menu is
  wanted, but the shipped corpus is faithful to the source recipe.
- Prompts are long: ~182 sentences mean ID, ~360 in the 41-50 band, so
  cells need MAX_CHUNKS >= 224 (ID) / >= 416 (longest band).

## 2026-08-13 (cont. 3): FSA-deduction domain ported (replaces ProofWriter as logic domain)

- Source located: alex:/home/hpc/c107fa/c107fa12/synthetic-RLVL (NOT ~/RLVL,
  which is an LLM-distillation pipeline). Generator: synthetic_dataset.py
  LogicDatasetGenerator, family hard_fsa_schema; single length knob `depth`
  (deduction layers); their bands are defined RELATIVE to the training max
  (band_train step<=train_max, band_ood >, band_hard_tail >=15) — reused.
  Paired natural-language and FOL renderings exist; we ported the NL form.
  Read-only w.r.t. the source repo; re-implemented rather than copied.
- Port (commit 5e3be12): src/textjepa/data/fsa_deduction.py emits a ground
  Datalog theory reusing ProofWriter's Fact/ProofRule types, so action
  phrasing, outcomes and the EXECUTOR are shared — no new evaluator.
  Counterfactuals carry teacher_rollout_actions (verified
  horizon_ranking_active=true), the field ProofWriter omits.
  configs/data/fsa_deduction.yaml + cell-script cases + 7 tests.
- Corpus data/intent_phrase/fsa_deduction (579 MB, MANIFEST): train 2000 /
  val 300 / test_id 200 at depth 15-26 (mean 40 inference steps), OOD
  bands 27-32 (57.8 steps), 33-40 (72.6), 41-50 (89.8), 200 each.
  108k transitions, 325k counterfactuals, disjoint seed spaces + content
  dedupe. Validation: expert-in-non-oracle-catalogue 1.0, exact replay
  1.0, goal success 1.0, deterministic regeneration byte-identical.
- TWO HONEST CAVEATS: (1) the feasible menu is DEGENERATE by construction
  (exactly one rule applicable per step, from the source recipe), so
  feasible_menu is trivially 100% here and the meaningful interface is the
  full non-oracle catalogue; a --side-facts-per-step knob can widen it if
  we want menu comparability with iGSM. (2) prompts are long (166
  sentences ID, 365 at depth 41-50) -> MAX_CHUNKS 224 (ID) to 416 (tail),
  i.e. expensive.
- NOT LAUNCHED: iGSM finishes first per the master plan.
- Grünau dispatcher live on the iGSM-hard round (scripts/gruenau_dispatcher.sh):
  claims PENDING cells onto genuinely idle GPUs (low mem AND low util).
  Sentence-LM cells needed model.max_chunk_len=96 (faithful sentences
  exceed the 48 default) — fixed and requeued.
- FSA side-facts variant built: data/intent_phrase/fsa_deduction_menu
  (side_facts_per_step=3, same 6 bands/counts, 1.2 GB) +
  configs/data/fsa_deduction_menu.yaml. Measured via the executor:
  feasible menu goes from mean 1.00 (min 1, max 1 — degenerate) to mean
  3.17 (range 1-7); full catalogue 159 -> 103. The feasible-menu protocol
  is now comparable across domains. Faithful 0-side-fact corpus retained.
- Owner measurement policy recorded in IGSM_MASTER_PLAN.md: run ALL
  interfaces (feasible_menu, full_catalogue, ldad_cycle, codebook_ground,
  autonomous) on EVERY checkpoint and band; choose main-text presentation
  at write-up. Marginal cost is eval only — one training run serves all.
- AUTONOMOUS interface implemented (master-plan task #27,
  `candidate_interface=autonomous`, needs `state_decoder=<decoder.pt>`):
  codebook proposals + environment-side NN grounding + endpoint-Energy
  beam search, and the DETACHED frozen-state decoder writes each step's
  text, which is appended to the model's own context and re-encoded. No
  menu, no oracle executor; ground truth scores the final answer only.
  `src/textjepa/planning/autonomous.py`, tests/test_autonomous_rollout.py.
  SMOKE (stab-ldad-ema-s0-v1 + decoder-s0-v1, 25 val problems, CPU,
  slack 2, k=256, top-k 8): answer_accuracy .04 (= 1/23 chance),
  well-formed .89, action-match .89, value-correct .08, first divergence
  at step 1, completion-claim .96 after 1.5 steps. Matches the
  preregistered expectation: structure survives, arithmetic does not, and
  the rollout jumps straight to the queried variable because nothing
  enforces prerequisite order once the executor is gone. Faithful iGSM is
  NOT supported yet (needs its own step decoder/parser) — plan.py raises.
- Autonomous interface now carries the LDAD cycle-consistency FEASIBILITY
  GATE (`autonomous_feasibility_gate=true` by default; calibration
  `midpoint` = halfway between the mean cycle score of feasible and of
  infeasible actions on 32 TRAINING problems walked along random feasible
  trajectories; labels used at calibration only, threshold -4.22 here).
  Matched smoke (same ckpt/decoder, 15 val problems, CPU, slack 2, k=256,
  top-k 8), gated vs ungated: answer accuracy .067 vs .067 (chance),
  value-correct .18 vs .13, mean steps 2.2 vs 1.6 (necessary 4.3),
  completion-claim .73 vs .93, stall .27 vs .00, gate pass rate .63.
  Reading: the gate does stop some of the "jump straight to the query"
  shortcut (claims drop, more prerequisite steps get written, the rest
  stall instead of faking an answer) but does not fix ordering — the
  query action still often passes the gate — and the arithmetic stays
  broken exactly as predicted. Ungated stays available as the ablation.

## 2026-08-13 (cont. 4): autonomous mode built + measured (task #27 closed)

- Report: research/reports/intent_phrase/2026-08-13-autonomous-mode/
  (mirrored). candidate_interface=autonomous (1f5e644 + gate ca310d7),
  24 tests. Final-answer accuracy at CHANCE (.067 on 15, .040 on 25;
  chance .043) — preregistration confirmed: ~90% of self-generated
  sentences well-formed and about the right variable, only 8-18% carry
  the right number, first divergence at step 1. Numbers get STUCK across
  steps ("is 9 times 8 = 9", "is 9 plus 0 = 9", "is 9").
- SECOND FINDING (new): with no executor, nothing enforces prerequisite
  order — endpoint Energy rewards reaching the goal and nothing refuses an
  infeasible action, so the planner writes the queried variable's sentence
  after ~1.6 of 4.3 needed steps. The environment had been silently
  supplying the feasibility gate.
- FIX APPLIED, partial: LDAD cycle-consistency feasibility GATE (our own
  AUC-.94 mechanism) thresholded before the Energy search, at roots and
  expansions; threshold calibrated on TRAINING problems only (midpoint of
  feasible/infeasible score means = -4.22; a 10th-percentile variant was
  nearly vacuous at 82% pass). Rejects ~37% of proposals: steps 1.6->2.2,
  immediate claims .93->.73, 27% of episodes now stall instead of
  fabricating an answer. Outcome UNCHANGED (.067) — the gate fixes the
  process, not the arithmetic. One scalar threshold on a .94-AUC score
  cannot enforce a dependency order.
- Paper reading: supports "better plans, rendering is the bottleneck", and
  isolates a second requirement for executor-free reasoning (feasibility/
  termination discipline). Smoke only — paper rows come from the 5-seed
  budget checkpoints. Faithful iGSM autonomous needs its own decoder+parser.
- ID BAND SATURATES AT BUDGET (157M, faithful iGSM-hard, op<=21): token LM
  strict .310 / s16 1.000; sentence LM .230 / .993; GoalHead .217-.247 /
  .977-.993 AND FLAT IN DEPTH (d1 .217 -> d16 .220). The in-distribution
  band cannot separate methods at this size — the paper's comparison must
  live in the OOD band, which is exactly why their train/test split was
  adopted.
- Eval-override gap found and fixed (e023f21): apply_eval_data_overrides
  exposed only stylized knobs, so the faithful OOD band (op 28-32) was NOT
  EXPRESSIBLE. Added eval_op_range/eval_max_op/eval_max_edge + plan.yaml
  keys; verified. Round 2026-08-13-igsm-hard-ood-v1 created (ID op 3-21 vs
  OOD op 28-32, 200 episodes, slack curve to 16) for every completed
  checkpoint, on its own dispatcher.

## 2026-08-13 (cont. 5): CRITICAL — feasible_menu is near-degenerate on faithful iGSM

First ID-vs-OOD rows looked BACKWARDS (GoalHead strict .265 ID vs .555
OOD). Checking baselines exposed the cause — it is a protocol artifact,
not a model result:

| band | planner | random | distractor rate | mean necessary |
|---|---|---|---|---|
| ID op3-21   | .265 strict / .995 s16 | **.245 / .985** | .351 | 7.1 |
| OOD op28-32 | .555 strict / 1.000 s16 | **.560 / 1.000** | .120 | 15.4 |

RANDOM MATCHES THE PLANNER ON BOTH BANDS, and beats it on OOD. Diagnosis
(measured on the generator): the faithful feasible menu is TINY — mean
3.41 actions (median 3) on ID, 3.01 (median 2) on OOD — while the fraction
of variables that are NECESSARY rises from 54% (ID) to 79% (OOD). So on
longer problems a random feasible pick is usually on the necessary path,
and "strict success" gets EASIER as problems get harder. Two consequences:

1. **Strict success is not comparable across bands.** Any ID-vs-OOD table
   using it is misleading; the whole slack curve plus the random/
   first-feasible reference lines must be reported per band.
2. **feasible_menu cannot discriminate methods on faithful iGSM at this
   budget** — same failure mode as FSA-deduction's degenerate menu, for
   the same structural reason (a small menu with a high necessary
   fraction). The discriminative interfaces here are full_catalogue
   (no-oracle) and the menu-free ones.

ACTION: the OOD round must be re-scoped to the full_catalogue and
menu-free interfaces, with random/first-feasible reference lines on every
band; feasible_menu is kept only as a saturated reference column. This
vindicates the "measure all interfaces" policy — a feasible-menu-only
headline would have been an artifact.
- OOD round v1 rows (feasible_menu, for the record only — see the
  degeneracy note above): sentence LM s4 .695 ID -> .835 OOD; token LM
  .735 ID; GoalHead .655 ID -> .800 OOD with random at .605/.810. The
  apparent "OOD is easier" holds for every row, confirming it is the
  protocol, not a model property.
- full_catalogue cells FAILED first attempt: the planner guard refuses
  lookahead>1 without allow_oracle_future_actions (correct — symbolic
  future menus at D>1 are candidate-privileged per the contract).
  Rewritten to measure BOTH per band: lookahead=1 with the flag OFF
  (fully oracle-free) and lookahead=16 with it ON (labeled
  candidate-privileged), so the paper can report the honest pair.
- OUR METHOD on the ID band with feasible_menu (hard-ldad, seed 0):
  lr 3e-4 s4 .620-.673 across depths, lr 1e-4 s4 .543-.653 — against
  RANDOM at .587. So LDAD is at or barely above random here, and the
  DEPTH AXIS IS FLAT (3e-4: d1 .660, d4 .673, d16 .620). This is the
  degeneracy again, now measured on our own method: with a ~3-action menu
  where 54-79% of variables are necessary, there is nothing for planning
  to add. Contrast stylized iGSM, where the same recipe scales .202 ->
  .975 across depth. The difference is a DOMAIN/PROTOCOL property, not a
  model regression — but it means faithful-iGSM feasible_menu must not be
  a headline table. full_catalogue cell for hard-ldad-lr3e4 queued.

## 2026-08-13 (cont. 6): full_catalogue is SILENTLY IGNORED on faithful iGSM

The "no-oracle full_catalogue" rows are NOT no-oracle. Evidence: they are
numerically identical to the feasible_menu rows and invalid_action_rate is
exactly 0.000 everywhere — impossible if a full catalogue were being
scored. Cause: scripts/plan.py dispatches data.name=igsm_real to
FaithfulPlanner, whose constructor takes only (lookahead, max_expand,
allow_oracle_future_actions) and whose _sequences() always calls
env.feasible_actions(). `candidate_interface` is never passed and never
read on this path, so every faithful eval — including all of today's —
has been a FEASIBLE-MENU eval regardless of the flag.

Same defect family as the earlier faithful silent drops (shuffle_actions,
the zeroed horizon-ranking loss, missing eval op_range): the faithful path
accepts a flag and ignores it. Consequences:
- Every faithful "full_catalogue" number recorded today must be relabelled
  feasible_menu and re-measured once the interface exists.
- The no-oracle interface DOES NOT EXIST for faithful iGSM; it must be
  implemented in FaithfulPlanner (candidate enumeration over the problem's
  whole variable catalogue + invalid-action handling), mirroring what
  ObservedActionPlanner/LatentPlanner already do.
- The load-bearing measurement for this domain is therefore still MISSING.
  Priority: implement it, since feasible_menu is degenerate here (LDAD
  .635-.640 vs random .605; OOD .820 vs .810).

## 2026-08-13 (cont. 7): full_catalogue IMPLEMENTED for faithful iGSM (5e7a5b7)

- FaithfulPlanner now takes `candidate_interface` + `invalid_action_mode`.
  `full_catalogue` scores the problem's WHOLE action catalogue (all of
  `FaithfulProblem.action_order`, no feasibility filter, resolved variables
  included — same shape as `range(len(problem.vars))` in the stylized
  planner); infeasible picks execute as no-ops via `step_or_invalid` and are
  counted in `invalid_action_rate`. Deeper lookahead is allowed WITHOUT the
  oracle flag on this interface (catalogue expansions consult no reference
  env); the oracle guard now applies to `feasible_menu` only. Unknown
  interfaces RAISE in both plan.py and the planner.
  `evaluate_faithful_planning` also emits `first_feasible_policy`, and both
  reference policies now run on the SAME interface as the planner.
  `feasible_menu` is bit-identical to the pre-fix code (verified against
  commit 3ddac49 and pinned as golden values in
  tests/test_faithful_candidate_interface.py, 8 tests).
- SMOKE (hard-ldad-lr3e4-s0-v1/best.pt, 30 val episodes, depth 1, slack 4):
  | interface | planner | random | first-cand | planner invalid |
  |---|---|---|---|---|
  | feasible_menu  | .567 | .467 | .533 | .000 |
  | full_catalogue | **.000** | .067 | .000 | **.936** |
  So the interface is now real and highly discriminative — and our
  feasible-menu-trained checkpoint FAILS it: it proposes an infeasible
  action 94% of the time, worse than uniform random over the same catalogue
  (71%), and never solves an episode. Menu-trained Energy has learned no
  feasibility signal on faithful iGSM. Every earlier faithful
  "full_catalogue" row must be relabelled feasible_menu.
- FOLLOW-UP: attempted-action MASKING added (`mask_attempted`, default true).
  Under invalid=noop an invalid pick leaves the state unchanged, so a
  deterministic argmin re-proposes it forever; masking the policy's OWN
  attempted actions (self-knowledge, same as `executed` in search.py /
  codebook.py) removes that artifact. Applied identically to the planner and
  to both reference policies. Unmasked stays measurable as an ablation.
  Re-run (same 30 episodes, depth 1, slack 4, seed 321):

  | interface | planner succ / inval | random | first-cand |
  |---|---|---|---|
  | feasible_menu           | .567 / .000 | .467 / .000 | .533 / .000 |
  | full_catalogue unmasked | .000 / .936 | .067 / .710 | .000 / .966 |
  | full_catalogue masked   | .033 / .674 | .033 / .626 | .033 / .642 |

  So the lock-in explained most of the .936 invalid rate, but the CONCLUSION
  HOLDS: with masking our planner is at random on both success (.033 = .033)
  and slightly WORSE on invalid rate (.674 vs .626), i.e. menu-trained Energy
  carries no usable feasibility signal on faithful iGSM. Caveat for the
  report: the budget (necessary+4 ~ 9.8 steps) is smaller than the catalogue
  (~15 actions), so even a perfect-recall random walk cannot finish; the
  honest comparison is planner-vs-random within band, not absolute success.
- MASKING FIX (adb19cd) + 3-way smoke on hard-ldad-lr3e4-s0 (30 eps, d1,
  slack 4). Attempted-action masking applied to planner AND both reference
  policies identically:

  | interface | planner succ/inv | random succ/inv | first-cand succ/inv |
  |---|---|---|---|
  | feasible_menu       | .567/.000 | .467/.000 | .533/.000 |
  | full_catalogue raw  | .000/.936 | .067/.710 | .000/.966 |
  | full_catalogue mask | .033/.674 | .033/.626 | .033/.642 |

  Lock-in was real and masking removes it (deterministic control .966 ->
  .642 invalid — the cleanest evidence, since it can ONLY lock in). But
  the conclusion survives: masked, the planner equals random on success
  (.033) and is slightly WORSE on invalid rate (.674 vs .626). Menu-trained
  Energy carries no usable feasibility signal on faithful iGSM. This is a
  model finding, not a harness artifact.
- BUDGET CAVEAT for any full_catalogue row: budget (necessary + 4 ~ 9.8)
  is smaller than the catalogue (~15), so even perfect-recall random cannot
  finish; absolute success is budget-limited. Honest comparison is
  planner-vs-random within band; for a non-degenerate absolute number the
  full_catalogue rows need slack extended past catalogue size.
- CONSEQUENCE: on faithful iGSM the model has feasibility knowledge only
  where the menu supplies it. The stylized answer to this was ldad_cycle
  (recover feasibility from the trained dynamics, AUC .94). ldad_cycle is
  NOT implemented on the faithful path either — that is the next build
  item, and it is the real test of whether the mechanism transfers.

## 2026-08-14: overnight audit — queue starvation, void fullcat rows, saturated ruler

**Ops (fixed).** `scripts/gruenau_dispatcher.sh` had wedged on a hung `ssh`
child at 05:05: the backgrounded remote job held the channel open, so ssh
never returned and the dispatcher blocked inside the launch. Effect: for
~14 h only 1 of 4 free cards was used while 9 cells sat PENDING. Worse, the
`timeout`-based patch made ssh exit 124 *after a successful launch*, so the
cell was "released" back to PENDING and re-launched — **5 concurrent copies of
`hard-sent-lat-lr1e4-s0-v1` were writing into one run directory**. Fix: fire
the ssh and then ask the CELL (job.sh writes RUNNING first) instead of
trusting ssh's exit code; `-n` + all remote fds redirected + `setsid`.
Duplicates killed, that cell reset to PENDING and relaunched clean; now 3
cells running, one per card. NEED_MB lowered 32000 -> 30800 (margin 800) so
gruenau1's 32 GB cards qualify at all (measured footprint ~30.6 GB).
NOTE: never `pkill -f` on gruenau1 *or over ssh to gruenau1* — this session
runs there; it kills our own shell (exit 144), hit twice again today.

**The 12 `fullcat_*` JSONs in 2026-08-13-igsm-hard-ood-v1 are VOID.** They ran
from snapshot `e023f21`, which predates the interface fix `5e7a5b7`, so
`candidate_interface` was still ignored. Proof: `invalid_action_rate` exactly
0.000 and no `first_feasible_policy` key. They must be relabelled
feasible_menu and re-measured from a post-fix snapshot.

**LR screen at the frozen 157M budget (8/18 cells), feasible menu, 300 eps,
slack-4 / slack-16 success:**

| row | d1 | d4 | d16 | slack16 |
|---|---|---|---|---|
| token LM (loops 1) | .740 | — | — | 1.000 |
| sentence LM (decoder) | .677 | — | — | .993 |
| goal-head 3e-4 | .680 | .620 | .650 | .990 |
| LDAD 3e-4 | .660 | .673 | .620 | .983 |
| TD-JEPA 3e-4 | .670 | .623 | .627 | .973 |
| LDAD 1e-4 | .653 | .543 | .557 | .960 |
| LDAD 1e-3 | .637 | .527 | .523 | .947 |
| goal-head 1e-4 | .617 | .600 | .613 | .983 |
| **random** | **.587** | .587 | .587 | **.980** |

Readings, all uncomfortable and all about the PROTOCOL, not the models:
1. **The ID band is short.** `mean_necessary = 7.1`, not the 15-25 we designed
   for — `op_range=[3,21]` yields ~7 necessary steps. So slack 16 is more than
   twice the whole solution length and random hits .980. Slack 4 is the only
   informative column, and there the spread over random is 5-15 points.
2. **LMs currently WIN.** Token LM .740 and sentence LM .677 sit at or above
   every JEPA row. On feasible-menu faithful iGSM we have no advantage to show.
3. **Depth does not help** (LDAD .660/.673/.620 across depth 1/4/16) — the
   near-degenerate ~3-action menu leaves nothing for lookahead to do.
4. **"OOD" is easier than ID**: op 28-32 (necessary 15.4) scores .81 vs .605
   ID at slack 4, for planner AND random alike. A fixed slack of 4 is
   proportionally far more generous on a 15-step problem than a 7-step one, so
   as designed this contrast measures the slack-to-length ratio, not length
   generalization. The ruler must be proportional (or ratio-matched) before any
   OOD claim.
5. Sanity check on the objective: the energy IS trained
   (`geo_horizon_rank=0.554` at epoch 9, `goal_dist_corr` -0.12 -> 0.63,
   `observed_action_sequence_exact=0.995`). Flag to verify: the recipe is named
   `mix4_aux025_nohorizon` yet the active ranking term is the HORIZON one
   (`geo_rank=0`). Naming or wiring needs a check before it goes in a paper.

**Consequence for the plan.** Faithful iGSM under a feasible menu is not a
measurement instrument at this length: saturated, flat in depth, and won by
the LM baselines. The load-bearing rows are the menu-free ones, and the two
blockers are now (a) re-run full_catalogue from a post-fix snapshot,
(b) implement `ldad_cycle` on the faithful planner (task #28), plus
(c) a proportional slack ruler and (d) a longer ID band.

## 2026-08-14 (cont.): random-at-.81 mechanism PINNED (fresh generator probe + sim)

Band structure (official generator, 20 problems/band): ID op3-21 —
catalogue 14.1, necessary 6.8 (49%), menu 3.75 (41% necessary). OOD op28-32 —
catalogue 21.6, necessary 16.1 (75%), menu 3.71 (64% necessary). A distractor
pick wastes exactly one step and never repeats (defined params leave the
menu), so random-menu failure = P(> slack wasted picks). Independent sim of
random-on-menu at slack 4 (60 problems/band): ID .633 (mean wasted 3.20),
OOD .800 (wasted 2.18) — reproduces the recorded .605/.81 within noise.
So OOD > ID for EVERYONE because op_range=[28,32] butts against max_op=32:
near the cap almost every parameter is an ancestor of the query, leaving ~5
distractors total against a fixed slack of 4. Not a code bug; the menu
protocol + their band placement makes the environment a near-oracle. The
menu setting needs max_op/max_edge well above the op band (more distractors)
plus proportional slack (#30) to discriminate at all; the load-bearing rows
remain the menu-free interfaces (#28, #29).

## 2026-08-14 (cont. 2): waterproofed protocol built end-to-end; instrument understood

- PROTOCOL FIXES (e2ca466): `necessary_range` banding (solution length, not
  n_op) + `slack_frac` proportional ruler; bands must also widen
  eval_op_range to their caps. Fixed bands: ID nec 8-15 @ caps 32/40,
  OOD nec 20-28 @ caps 48/64.
- MASK FIX (8acd62c): the permanent attempted-mask made any necessary action
  tried too early unrecoverable — full_catalogue success was 0.000 for EVERY
  policy. Mask now scoped to the current state (resolved + invalid attempts
  since last progress, reset on progress). 13 tests incl. new recoverability
  regression.
- SMOKE (hard-ldad-lr3e4 ckpt, 100 eps/cell, gruenau2):
  * feasible_menu, slack_frac 0.5: planner .71/.87 (ID/OOD), random .62/.86,
    first-cand .70/.87; frac 0.25: .56/.67 vs random .55/.64. OOD inverts
    AGAIN and provably unfixably: menu waste is bounded by the distractor
    count (unrepeatable picks), which the proportional cushion exceeds on
    long problems. Feasible menu = appendix-only near-oracle control.
  * full_catalogue, corrected mask, slack_frac 0.5: SUCCESS 0.000 for all
    three policies (planner invalid .820/.891, random .761/.834, first-cand
    .813/.893). Understood: every attempt consumes budget, and blind search
    needs ~ necessary x (catalogue/menu) ~ 5x necessary attempts; budget 1.5x
    cannot suffice. The instrument is therefore success-vs-ATTEMPT-BUDGET
    (multiples of necessary); a policy with feasibility knowledge approaches
    1x, blind search needs ~5x. Generous-budget variant (frac 4.0) launched
    to verify random>0.
  * MODEL FINDING confirmed on the fixed protocol: planner invalid rate is
    WORSE than random (.82 vs .76 ID; .89 vs .83 OOD) — menu-trained Energy
    still carries no feasibility signal; ldad_cycle port (#28) is the
    critical path.
- REPORTING DECISION (with owner, this session): main results = menu-free
  (full_catalogue budget curves + codebook_ground/ldad_cycle/autonomous vs
  free LM generation); feasible_menu demoted to labeled appendix control.

## 2026-08-14 (cont. 3): budget-curve instrument VERIFIED on full_catalogue

At budget 5x necessary (slack_frac 4.0, 100 eps, hard-ldad-lr3e4 ckpt):
ID nec8-15: planner .530 (inv .793), random .700 (inv .700), first-cand .510.
OOD nec20-28: planner .140 (inv .882), random .480 (inv .796), first-cand .100.
Correct ordering restored (OOD harder for everyone), success mid-range,
policies separated. Instrument for the paper: success vs attempt budget in
multiples of necessary. Planner sits BELOW random on both bands — the
menu-trained Energy prefers infeasible actions it never had to reject; this
is precisely the headroom the ldad_cycle port (#28) must fill (stylized
feasibility AUC .94). Narrative: menu-trained energy fails menu-free ->
cycle-consistency recovers feasibility -> autonomous operation.
