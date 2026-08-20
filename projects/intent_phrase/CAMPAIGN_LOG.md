# Intent-phrase campaign log (living handoff)

_Keep short. Compress completed stages into a few lines; details live in
`research/reports/intent_phrase/<date>-*/REPORT.md` and are mirrored to
`/vol/home-vol2/ml/laitenbf/TextJEPA-paper/reports/`. Last update: 2026-08-20._

## ARCHIVED: 2026-08-07 .. 2026-08-16 (full text in a dated report)

Everything from those ten days now lives verbatim in
`research/reports/intent_phrase/2026-08-20-campaign-archive-through-0816/REPORT.md`.
What still matters from it, in one place:

- **Environments.** Stylized iGSM, faithful iGSM (the paper's generator),
  plus FSA-deduction, PlanBench Blocksworld and text-ALFWorld as extra
  domains. Model size fixed from the iGSM paper (GPT2-small shape).
- **Three interface modes** used throughout: `feasible_menu` (environment
  supplies legal actions), `full_catalogue` (every action in the problem,
  legality is the model's job), and menu-free (`prior_propose`,
  `codebook_ground`, `ldad_cycle`, `autonomous`).
- **Reference-policy warnings that still bind.** `feasible_menu` is
  near-degenerate on faithful iGSM and on iGSM-med (random scores .98/.59),
  so it is a reference column only, never a headline. On stylized use
  *random*, not first-feasible, as the menu-free reference — catalogue order
  is near-topological and first-feasible is therefore not a fair baseline.
  `full_catalogue` was silently ignored on faithful iGSM until 5e7a5b7;
  results predating that commit are void.
- **Menu-free on faithful iGSM was the open decision** entering 08-17:
  cycle-consistency feasibility is stylized-only (AUC .85 stylized vs .48,
  chance, on faithful through the same code path); easy-negative contrast
  training did not fix it; hard negatives were the only rising curve at .570
  — real but weak. Superseded by the 08-19/08-20 flat-backbone work below.
- **Probe caution established early** (08-12): the feasibility signal is not
  JEPA-specific — LM states carry it too. Re-confirmed with random-init
  controls on 08-20.

## 2026-08-17 — owner redirect: budgets out, LMs must work, push hard negatives
- Owner decision (chat): (1) retire scored attempt budgets — primary instrument
  becomes accuracy + distribution of steps used (run until solved or runaway
  cap); (2) token/sentence LM MUST work on faithful iGSM (original iGSM paper
  did) — suspected harness fault: we evaluated LMs by candidate-ranking under
  budget instead of free generation; (3) push hard-negative cf-contrast with
  more counterfactual data, parallel sweep.
- LAUNCHED `2026-08-17-cf-k-sweep-v1`: hardneg-k{4,16,64,128}, weight 4,
  unresolved-only, effective batch 4 via +train.microbatch_size grad-accum,
  snapshot a26f28e. Hosts: k4 gruenau7:0, k16 gruenau7:2, k64 gruenau9:0
  (A100-80), k128 gruenau10:0 (A100-80). Each job.sh runs the cycle-AUC probe
  on best.pt at end → cycle_auc_probe.json. Wiring of
  data.invalid_counterfactual_k verified (faithful.py:256/300/433,
  checkpoint.py:292-295).
- IN FLIGHT: free-generation LM eval port (plan_lm.py mode) + train/eval
  mismatch diagnosis; round `2026-08-17-lm-freegen-v1` expected.
- 2026-08-17 LM freegen verdict (round `2026-08-17-lm-freegen-v1`, commits
  ef7a732/3e38210): free-generation eval ported (greedy decode, env-grounded,
  goal-reached grading, runaway cap 4x nec, no scored budget). Harness
  VERIFIED correct (5-episode dumps: format/prompt/parse all match training).
  tokLM & sentLM free-gen success = .000 (ID band, 200 eps, both invalid
  policies); teacher-forced on TRAIN traces: greedy next intent only 32%
  feasible. Old candidate-ranking numbers were propping LMs UP. Root cause:
  our LM row was NOT the Ye et al. recipe — loss on intent phrases only, no
  outcome/arithmetic text, trained caps 21/28 vs eval 32/40. Also ~1.5% vocab
  OOV on catalogue names (minor).
- LAUNCHED `2026-08-17-lm-fullsol-v1`: proper Ye-et-al-style token LM — CE on
  full solution text, caps 32/40/[3,32] nec 8-15 — with automatic free-gen
  eval at end. This is the "LMs must work" fix.
- `2026-08-17-lm-fullsol-v1/tok-lm-fullsol-caps32-lr3e4-s0` RUNNING on
  gruenau10:2 (A100): Ye-style full-solution CE (commit 5d99a99,
  train.lm_loss_on=all_solution, wiring smoke-verified), caps 32/40/[3,32]
  nec 8-15, new cap-keyed vocab (faithful_vocab_32_40.txt, OOV .012%).
  ~7 h train, then auto free-gen eval (gen_outcome=model: LM writes its own
  arithmetic, only definition sentences stepped; grades success_rate and
  success_answer_rate). Old checkpoints/vocab untouched.
- 2026-08-17 iGSM-faithfulness AUDIT (agent, full findings in chat/report):
  generator+rendering genuinely faithful (vendored official iGSM, mod-23,
  reference renderer, machine-checked). THREE suspect deviations: (1) eval
  band op<=32/edge40/nec8-15 is AT/BEYOND the paper's hardest OOD point for a
  FULLY-trained model — we eval an undertrained model exclusively there;
  (2) token budget ~0.1B tokens vs paper's full pretraining (100-1000x under);
  (3) LM row trained intent-only loss (~10% of solution tokens) vs paper's
  full-solution CE. Minor: no hash-bin train/test dedup (paper uses hash<17
  vs >=17 mod 23) — reviewer-proofing gap; frozen 21/28 vocab corrupts ~20%
  of 32/40-cap problems (OOV names). 15% distractor injection judged mild.
- LAUNCHED (agent): (A) in-dist freegen eval of existing tokLM at its own
  training caps 21/28 no nec-filter → cell
  2026-08-17-lm-freegen-v1/tok-lm-freegen-indist-s0; (B) paper-repro run
  2026-08-17-lm-med-repro-v1/tok-lm-med-fullsol-s0: iGSM-med caps 15/20,
  all_solution loss, 20x30k epochs, freegen_curve.jsonl every 2 epochs
  (success-vs-tokens slope = tests the undertraining hypothesis).
- 2026-08-17 JEPA ARCHITECTURE AUDIT — SMOKING GUN. Measured on the hardneg
  checkpoint: (1) recipe trains the LEGACY Markov MLP predictor
  (predictor_kind=concat via paper_recovery_mlp_geometry.yaml inheritance;
  allow_legacy_predictor=true silenced the train.py guard) — the causal
  history predictor never trained; (2) predictor output scale exploded ~2e6
  (imagined displacement norm 1.5e7 vs real 7.2; every touching loss is
  scale-invariant) → raw cos(imagined,real)=-0.001, so ALL cycle/manifold
  diagnostics and planning scores were reading noise; (3) latent_pred = <1%
  of predictor gradient (cf_contrast 58%, ranking 24%) → predictor worse
  than identity in LN space and action-blind (cos .96 under action
  permutation; d_action=16 pooled). Encoder causality, EMA, detaches, term
  plumbing all verified CORRECT. This retroactively explains the off-manifold
  finding, chance cycle AUC, and possibly the whole faithful menu-free
  failure. Full audit in chat; script in session scratchpad.
- ACTION: cf-k-sweep-v1 (legacy predictor → wasted) KILLED; relaunching as
  `2026-08-17-causal-fix-v1`: causal-k0 (fix only, no contrast — isolates the
  bug), causal-k8, causal-k64, causal-k8-act64 (d_action 16→64), all with
  latent_pred reweighted to first-class predictor gradient and auc_curve
  probes every 2 epochs.
- 2026-08-17 in-dist freegen eval (tok-lm-freegen-indist-s0, gruenau2:1):
  existing intent-only-CE tokLM at ITS OWN caps 21/28, no nec filter —
  success .010 (fail) / .020 (ignore), first step invalid in 99% of episodes.
  So the OOD eval band was NOT the main LM problem; the intent-only training
  loss is. Ranking of suspects updated accordingly.
- `2026-08-17-lm-med-repro-v1/tok-lm-med-fullsol-s0` RUNNING gruenau7:3
  (snapshot 5d99a99, no new commits needed): iGSM-med caps 15/20, no nec
  filter, distractor_prob=0, all_solution CE, 20x30k, gen_outcome=model
  evals; freegen_curve.jsonl every 2 epochs (first point ~3 h in), full eval
  at end; ETA ~32 h (~Aug 19 early).
- `2026-08-17-causal-fix-v1` RUNNING (commit 0f52367: cell script no longer
  hardcodes allow_legacy_predictor=true; guard verified by negative+positive
  dry-runs; latent_pred weight 15, predictor_kind=causal). Cells: causal-k0
  (fix only) g7:0, causal-k8 g7:2, causal-k64 g9:0 (mb1, may hit timeout),
  causal-k8-act64 (d_action 64, token_bottleneck) g10:0. Old sweep KILLED
  cleanly (PID-verified, incl. 29 orphaned dataloader workers). auc_curve
  .jsonl every ~2 epochs per cell; first points ~6-7 h in; ETA 26-35 h.
- lm-med-repro epoch-1 freegen point: success .000 but
  outcome_value_match .615 (arithmetic forming), first step invalid .79 —
  early; slope over next epochs is the signal.
- lm-fullsol-caps32 COMPLETED (~7 h + eval): full-solution CE at caps 32/40
  improves the pieces (first-step invalid .55 vs .76 intent-only,
  unparseable .05 vs .12, own-arithmetic match .76) but success still .000
  (fail) / .015 (ignore) — every episode hits an invalid step. Together with
  the FLAT med-repro curve (epochs 1-3), suspicion moves to an eval-harness
  format divergence or a residual training-signal gap; targeted diagnostic
  agent running (teacher-forced vs free-gen token-stream diff on the
  med-repro checkpoint).
- 2026-08-17 med-repro flat-curve diagnostic (agent, CPU on epoch-3 ckpt):
  harness VERIFIED correct end-to-end — eval prompt token ids byte-identical
  to training streams, loss mask covers all outcome tokens (41/41, 89/89),
  env feasible set at t=0 exactly the DAG sources, eval vocab = checkpoint
  vocab. Flat curve is real early-training behavior: val_loss .98→.59 and
  first-step-invalid .794→.746 ARE improving; success stays 0 under
  fail-policy until first-step feasibility clears ~13% random-plausible.
  Judge at epochs 7-11. Bug found+fixed (not the cause): names built with
  .replace("each ","") corrupted entity names containing "each" ("Beach
  Homes"→"BHomes"; 8/2337 params ungroundable) — commit c00526d, snapshot
  archived; rerun periodic evals from new snapshot for clean grading. Also
  noted: 15/20 vocab misses rare punctuation-attached forms (~20% of
  problems have 1-2 <unk> prompt tokens) — rescan if recipe kept.
- med-repro freegen curve (50-ep, fail-policy): success 0/.0/.0/.02/.04/.02/
  .06 at epochs 1-13; invalid-step .79→.42 monotone, unparseable .25→.08.
  Reading: recipe learns, budget short — plan a scaled run after epoch 20.
  causal-fix cells: causal predictor ~4x slower than MLP (~14 h/epoch);
  first AUC probes expected 2026-08-18/19.
- 2026-08-18 med-repro COMPLETED (20 epochs, 600k problems): final 200-ep
  free-gen at own caps 15/20: success .110 (fail) / .205 (ignore), invalid
  .34, unparseable .06 — up from ~0 for intent-only training. Recipe is
  RIGHT, budget short; curve still rising at end. Scaled run
  (lm-med-scale-v1, 100k/epoch x30, gruenau10:2) tracking: success .04 at
  epoch 3 (~300k problems seen).
- causal-k0 epoch-1 cycle AUC .483 (chance) — expected this early; k0 has no
  pred_cycle term, so cycle AUC can only rise via displacement becoming
  on-manifold. Judge at epochs 4-6.
- 2026-08-18 scaled LM (100k/ep): success .02/.04/.10/.18/.30 at epochs
  1/3/5/7/9, invalid-step .70→.19 — accelerating, first fully-own-arithmetic
  correct answers from epoch 7. TOKEN LM CONFIRMED LEARNABLE with paper
  recipe + data scale. causal-fix early AUCs: k0 .483, k8 .524,
  k8-act64 .557 at epoch 1 (old broken recipe needed 9 epochs for .570).

## CURRENT STATE (2026-08-18, evening) — "make it work" campaign
- BREAKTHROUGHS this cycle: (1) token LM works with the paper recipe
  (full-solution CE + fresh data scale): scaled run at 40% free-gen success
  by epoch 11/30 and accelerating (intent-only recipe: ~0%). (2) JEPA audit
  found the recipe trained the legacy Markov MLP predictor with exploded
  output scale and latent_pred at <1% of predictor gradient — all prior
  faithful menu-free negatives are confounded by this.
- RUNNING: scaled LM lr3e4 (g10:2, ~2 d left), LM lr1e3 (g7:3, ~3 d), LM
  lr1e4 pending GPU. JEPA causal-fix cells being relaunched from a faster
  snapshot (profiler agent in flight; old pace 14 h/epoch was ~4x too slow)
  as 2026-08-18-causal-fix-v2: causal-{k0,k8,k64,k8-act64} + MLP-control LR
  sweep mlp-k8-lr{1e3,3e4,1e4,3e5} (arch-vs-weighting isolation). All cells
  self-probe: freegen_curve.jsonl / auc_curve.jsonl every 2 epochs.
- DECISION RULES: JEPA cycle AUC must clear ~.8 before planning rows are
  worth rerunning. LM eval = free generation + steps-used distribution
  (budgets retired). Old candidate-ranking LM numbers are invalid as
  baselines (they propped weak LMs up).
- NEXT: read auc_curve/freegen_curve across cells; pick winning JEPA recipe;
  then scale winning recipes to caps 32/40 and port no-budget protocol to
  JEPA planning eval. Reports: research/reports/intent_phrase/
  2026-08-16-faithful-menufree-decision/ (superseded in parts by the
  2026-08-17 audits — see entries above).
- 2026-08-18 JEPA speedup (commits bf47186, ce23b53; snapshot ce23b53):
  compute-bound, not data-bound. Fixes: unique-chunk/pad-trim EMA teacher
  encoding (2.0s→0.03s, was re-encoding identical prefixes per candidate),
  bf16, vectorized counterfactual readout (bit-exact, 96 tests pass),
  k64 microbatch 1→4. Round relaunched as `2026-08-18-causal-fix-v2`:
  causal-{k0 g7:0, k8 g7:2, k64 g9:0 (~10x faster), k8-act64 g10:0} +
  MLP-control LR sweep mlp-k8-lr{3e4 g9:1, 1e3 g10:1, 1e4 g7:0p, 3e5 g7:2p}.
  ~1.2-1.7 h/epoch → 10 epochs in ~14-17 h. All 8 RUNNING, probes armed.
- 2026-08-18 Alex submissions (`2026-08-18-alex-lm-v1`, local README has all
  remote paths + pull-rsync command): lm-med lr1e4 (jobs 4040212-217) and
  tok-lm-hard32-fullsol-big-s0 — caps 32/40 x 100k x 30, the target-band LM
  scale-up (jobs 4040218-225). 24h wall limit handled via resume-chaining
  (train_lm_resume.py, Alex-side patch only). Runs live in Alex $HOME
  because the atuin $WORK group inode quota is exceeded (573K/500K, grace
  expired) — admin attention needed. Alex cannot ssh out; results pulled
  from Grünau side.

## 2026-08-19 readout (after ~1 day)
- TOKEN LM SOLVED at iGSM-med (caps 15/20, own caps, 200 eps, free-gen,
  model writes own arithmetic): lr3e-4 .810 strict / .840 lenient; lr1e-3
  .825 / .850; invalid-step .03. Curves: 0→.86 over 30x100k. LR 1e-3 ≈
  3e-4 (not LR-limited at this scale); lr1e-4 + caps-32 scale-up queued on
  Alex. CAVEAT: success_answer_rate only .06-.09 while per-step arithmetic
  match ~.9 — suspected final-answer grading bug, being checked.
- JEPA causal-fix-v2 (all 8 done, 10 epochs): cycle AUC trajectories flat/
  weak for ALL cells — causal-k8 .53, causal-k64 .53, causal-k8-act64 .57,
  mlp-k8 lr1e-3 .49 / 3e-4 .51 / 1e-4 .60 / 3e-5 .58. causal-k0 OOM'd
  (packed GPU) at epoch 3 (.50). VERDICT: correct architecture + loss
  weighting do NOT make LDAD cycle-consistency a feasibility signal on
  faithful; the causal predictor is not the limiter for this readout; LR
  1e-3 collapses. The cycle mechanism is closed as the menu-free route.
  With-menu slack-curve planning on these ckpts is healthy (.32→.79).
- NEXT (launching): feasibility through the ENERGY head (the target
  system's own mechanism): probe energy-head feasible-vs-infeasible AUC on
  v2 checkpoints; add self-supervised energy ranking term observed-next vs
  hard-negative counterfactual-next (predictor in loop); train on freed
  Grünau GPUs; port no-budget/steps-distribution eval to JEPA planning.

## 2026-08-19 — OWNER DECISIONS: paper plan + 1-week architecture push
- Record audit (this session): only feasible_menu ever worked; full_catalogue
  and no-menu (ldad_cycle .29 slack-4, codebook_ground .28 / recall 1.0,
  autonomous chance) never exceeded ~.3 on stylized and were <= random on
  faithful. "JEPA probes beat LM probes" is contradicted by 08-12 LM
  state-readout report — representation story must be consequence-based
  geometry / use, not probe superiority.
- PAPER PLAN (owner): analysis-heavy; floor = stylized feasible-menu JEPA vs
  token/sentence LM + depth scaling + looped-LM test-time-compute; emphasis
  = representation analysis (paraphrase clustering / negation separation,
  counterfactual geometry, LDAD decodability) across domains; stretch =
  faithful iGSM OOD in modes feasible/full-catalogue/no-menu. 1 week
  architecture push, then ~3 weeks finals.
- ARCHITECTURE DECISION (owner): intent JEPA moves to a FLAT TOKEN BACKBONE
  (causal token transformer, init from the trained token LM; state = hidden
  at intent boundaries; predictor + endpoint energy + LDAD + EMA unchanged;
  backbone-matched vs LMs). No-menu: generative intent-phrase prior head
  (CE on observed intents) proposes, JEPA energy plans; codebook kept as
  comparison. Energy hard-negative feasibility ranking to be added.
- Day-1 launches: (a) oracle-free depth-curve re-measure of stylized
  headline; (b) flat-backbone intent JEPA v2 build + iGSM-med smoke/train;
  (c) energy-head feasibility AUC probe on existing checkpoints.
- 2026-08-19 ENERGY-HEAD FEASIBILITY PROBE (commit b2d53ff, round
  2026-08-19-energy-auc-probe-v1): faithful ckpts energy AUC ~chance
  (.47-.51), like cycle. STYLIZED HEADLINE: cycle .846 but the planning
  ENERGY head is ANTI-feasible (.386 pooled / .318 per-state) — imagined
  successors of premature actions get LOWER energy. Energy = goal-progress,
  never legality.
- 2026-08-19 ORACLE-FREE DEPTH RE-MEASURE (round
  2026-08-19-oraclefree-depth-v1, 5 seeds x 300 eps, feasible menu at root,
  oracle-free expansion at depth>1):
  | depth | strict | slack2 | slack4 | OLD strict (symbolic future menus) |
  |   1   | .202   | .650   | .898   | .202 (identical — sanity check) |
  |   2   | .071   | .399   | .784   | .729 |
  |   4   | .013   | .205   | .641   | .885 |
  |   8   | .021   | .226   | .655   | .965 |
  |  16   | .019   | .215   | .649   | .975 |
  VERDICT: the headline depth scaling was ENTIRELY carried by symbolic
  future menus. Oracle-free, deeper search HURTS (energy prefers illegal
  imagined continuations, cf. anti-feasible AUC). Paper consequence: depth
  claims must be re-earned with a legality-aware energy; the v2
  energy_cf_feasibility_rank term (anchor + along imagined rollouts) is the
  central mechanism, with flat-lminit-nocfrank-s0 as its ablation.
- 2026-08-19 ALEX: existing chains healthy (lm-med lr1e4 ep10 freegen .26;
  hard32 caps-32 ep5 freegen .12 — the target band IS learning with scale).
  NEW round `2026-08-19-alex-lm-finals-v1` (snapshot 502d2ad, lr1e-3,
  100k x 30): tok-lm-hard21 s0/1/2 (paper split, ID + OOD 28-32), tok-lm-med
  s1/2, sent-lm-med s0, sent-lm-hard21 s0 (sentence LM gained all_solution
  decode loss). Jobs 4046275-4046292 PENDING. RISK: Alex $HOME 91/100G.
- 2026-08-19 FLAT INTENT JEPA v2 BUILT + LAUNCHED (docs/flat_intent_jepa_
  design.md; commits 985643b/a112ba8/51a4cb0/b60c9a7/4f8e99e; snapshot
  4f8e99e): one causal token transformer (init from tok-lm-med best.pt),
  s_t = hidden at outcome_t end, action = intent hidden in context via a
  block-attention phrase pass (all catalogue candidates in ONE forward),
  EMA teacher, residual-MLP predictor, horizon-blind energy, LDAD, tied LM
  head = intent prior. Losses: latent_pred 1, vicreg 1, cf_state 1,
  geo_horizon_rank 1, geo_adv_mse .25, ldad 1, energy_cf_feasibility_rank 4
  (anchor: 2 feasible + 8 premature + 4 resolved negatives; PLUS along every
  imagined rollout prefix, k=4), intent_prior_lm 1 (replaces chunk_pred).
  Planner scripts/plan_flat.py: feasible_menu / full_catalogue / ldad_cycle /
  codebook_ground / prior_propose (16 sampled phrases, grounded) /
  autonomous; no budget, steps-used distribution, oracle-free lookahead with
  imagined_invalid_rate. Round `2026-08-19-flat-jepa-v1`: flat-lminit-s0
  (g10:0), lr3e5 (g10:1), scratch (g10:2), nocfrank ablation (g7:0),
  frozen-encoder (g7:1, FAILED - to fix). ~.07 s/problem, ~1.9 h/epoch A100.
- 2026-08-19 FLAT-JEPA v2 EPOCH-0 CURVE (iGSM-med ID, 100 eps, no budget,
  runaway cap 4x; random = same-interface random reference):
  | interface | flat-lminit | ablation (cf-rank off) | random |
  | feasible_menu d1 | .99 | .99 | .98  (DEGENERATE at these caps) |
  | full_catalogue d1 | **.72** | .60 | .59 |
  | full_catalogue d4 | .48 | .51 | .59 |
  | prior_propose d1 | .56 (5.2 steps, 1.05x necessary, recall .91) | .44 | .59 (16-21 steps) |
  energy AUC depth0/depth1: lminit .606/.594, ablation .512/.493.
  FIRST EVER above-random full_catalogue (mode 2) result, and it is caused by
  energy_cf_feasibility_rank (ablation = random). prior_propose (mode 3)
  solves in ~1.05x necessary steps vs random's 3x — the steps-distribution
  instrument is the right frame for no-menu. Epoch 0 of 10; d4 still below
  d1 (deeper search not yet helped — watch as legality AUC rises).
- 2026-08-19 local hard21 token LM launched (gruenau1:2, snapshot 502d2ad,
  same recipe/snapshot as the Alex hard21 seeds → bit-comparable): caps
  21/28, vocab 3479 tokens, OOV .09%, 0.75 s/step, ~39 h (ETA 2026-08-21
  ~09:00 UTC). Hedge against the Alex a100 queue; will initialize the
  faithful-band FlatIntentJEPA. NOTE for future LM runs: scripts/train_lm.py
  has NO bf16/autocast (configs/lm.yaml precision key is dead) — adding it
  is a free ~2x on all LM training.
- 2026-08-19 v2 ROUND COMPLETE (12 cells running). iGSM-med round
  `2026-08-19-flat-jepa-v1`: s0/s1/s2 (seeds; s1/s2 also vary data seed),
  lr3e5, scratch, frozen, nocfrank (w0), ecf1 (w1), ecf16 (w16), noprior
  (intent_prior_lm=0) — dose-response for the central term = 0/1/4/16.
  NEW `2026-08-19-flat-jepa-stylized-v1/flat-stylized-s0` (g7:2, scratch,
  ~33 h) and `2026-08-19-flat-jepa-hard21-v1/flat-hard21-scratch-s0`
  (g7:3, caps 21/28, ~70 h shared-GPU; lminit sibling pending the hard21 LM).
  Stylized port (commit 4a56784, snapshot archived): IGSMDataset gained the
  hard-negative + rollout-counterfactual pools (defaults 0 = legacy
  bit-identical), new src/textjepa/data/stylized_flat.py adapter, shared
  build_eval_dataset; 11 tests pass; stylized vocab 108 tokens, hard21 vocab
  covers the 32/40 OOD band with 0% OOV.
  CAUTION for the paper: on stylized, `first_feasible` is privileged (the
  catalogue order is near-topological → 100% success); the honest menu-free
  reference on stylized is RANDOM.
- 2026-08-19 EPOCH-0 DOSE-RESPONSE for energy_cf_feasibility_rank (iGSM-med
  ID, 100 eps, oracle-free lookahead; random reference .59):
  | weight | energy AUC d0 / imagined-d1 | full_catalogue d1 | prior_propose d1 |
  |   0    | .512 / .493 | .60 (= random)  | .44 |
  |   4    | .606 / .594 | .72             | .56 |
  |  16    | **.771 / .852** | **.88**     | .64 |
  MONOTONE and strong: at weight 16 the energy separates legal from illegal
  continuations at AUC .85 along IMAGINED rollouts, and menu-of-all-actions
  planning hits .88 vs random .59 — the first decisive menu-free-ish result
  in the project. LAUNCHED weight 64 (flat-lminit-ecf64-s0, g10:2) to find
  the top of the curve. OPEN PUZZLE: depth 4 still < depth 1 everywhere
  (.54 vs .88 at w16) — deeper oracle-free search still not paying off;
  diagnose next (beam/expansion pool, not legality, is now the suspect).
  Also: lr3e-5 favours prior_propose (.69) over lr1e-4 (.56) while lr1e-4
  favours full_catalogue — worth a note. flat-scratch prior_propose = .00
  (no LM init → proposer cannot form phrases), confirming LM init is what
  makes the no-menu mode possible. noprior cell died on "too many open
  files" (dataloader) → relaunched with ulimit 65536 + 6 workers.

## 2026-08-19 (evening) — depth is the open problem; drift is the lead suspect
- FULL epoch-0 numbers, best cell flat-lminit-ecf16-s0 (energy_cf weight 16),
  iGSM-med ID, 100 episodes, no budget, oracle-free lookahead:
  | setting | depth 1 | depth 4 |
  | legal-moves list (feasible_menu) | .99, 8.5 steps (1.42x min), 0% illegal | .99, 10.0 steps |
  | full list (full_catalogue) | .88, 12.2 steps (1.71x), 30% illegal | .54, 20.1 steps, 62% illegal |
  | no list (prior_propose) | .64, 1.07x min steps when solved, 6% illegal, proposal recall .92 | .38, 16% illegal |
  RANDOM reference: .98 / .59 / .59. Legal-list setting is saturated
  (random .98) — reference column only, as on faithful.
- CORRECTION to an earlier chat claim: the "16-21 steps" figure quoted for
  the random policy was actually OUR model's step count under full_catalogue.
  The eval records steps only for our planner, NOT for the random/
  first-feasible references. ACTION: add step recording for reference
  policies to the watcher/eval (needed for the steps-distribution figure).
- No-list failures are "gives up", not "wanders": flat_search ends the
  episode when no usable proposal exists (n_no_proposal, break at
  flat_search.py:382); proposal recall .92 per state compounds over ~5 steps.
  Cheap fixes (eval-time, no retraining): more proposal samples per step
  (now 16) and/or fall back to the full catalogue when nothing is usable.
- DEPTH IS THE OPEN PROBLEM: deeper search hurts in both discriminative
  settings, and the illegal-pick rate roughly DOUBLES from depth 1 to 4
  (30->62%, 6->16%). Not a legality-signal problem any more (AUC .77 at the
  current state, .85 at imagined depth 1).
- DIAGNOSTIC RUNNING (agent): (i) OWNER'S 2x2 — {learned energy vs ORACLE
  latent-distance-to-solved-state} x {imagined endpoints vs truly-executed
  endpoints}, depths 1/2/4/8, 200 eps, all oracle rows labelled
  candidate-privileged. Reading: energy<oracle on imagined ⇒ the head is the
  limiter; imagined<executed ⇒ predictor drift is the limiter; oracle+
  executed still flat ⇒ the endpoint-distance formulation itself is wrong.
  (ii) DRIFT CURVE (prioritized): cosine/L2 between k-step imagined and true
  EMA states for k=1..8, against the typical distance between unrelated real
  states, in the layer-normalized space the loss uses, plus a check that the
  old 2e6 norm explosion has not returned, plus energy legality AUC at
  imagined depth 2/3/4.
- OWNER FIX LAUNCHED (agent): multi-step rollout prediction. The predictor is
  a cheap 2-layer residual MLP (model.predictor_kind=mlp) trained ONLY
  one step ahead, so imagined states plausibly leave the manifold after 1-2
  steps. New opt-in objective `latent_rollout_pred` regresses the k-step
  imagined state onto the EMA teacher's true state for k in [1,2,4] and
  [1,2,3,4,6,8], per-k losses logged plus a live cosine drift diagnostic.
  Cells flat-lminit-ecf16-roll124-s0 and -roll1248-s0 (energy_cf 16 + rollout
  1), evaluated at depths 1 AND 4 every 2 epochs.

- **Drift measured, and it is real** (commit 408fb0f, snapshot
  `runs/autonomy/_code/408fb0fd433a6cfe7a3e6e391372e6088210ede1/`). Cosine
  similarity between the k-step imagined state and the true EMA state, on
  the LM-init encoder before rollout training: k1 .607, k2 .571, k4 .470 —
  imagined futures decay with depth, which is the shape needed to explain
  why deeper search hurt. After 140 steps of `latent_rollout_pred`: k1 .860,
  k2 .857, k4 .860 — the decay flattens entirely. Root cause: the predictor
  was only ever trained one step ahead but asked at planning time to imagine
  four. Sanity check passed: the k=1 term equals `latent_pred` exactly at
  init; grads reach encoder (4.37e-1) and predictor (1.69e-1). No measurable
  throughput cost (~0.10 s/problem). Cells on gruenau10:0 and gruenau9:0,
  ~28-30 h for 10 epochs, first depth-1-vs-depth-4 watcher readout ~3 h in.

- **Seeds s1/s2 crashed at ~step 2000 (epoch 0)**, not a modelling failure:
  dataloader worker file-descriptor exhaustion (`FileNotFoundError` in
  `multiprocessing.resource_sharer`), same family as the earlier
  `Too many open files` kill. Their job.sh predated the fix (no
  `ulimit -n 65536`, `num_workers=12`). Old dirs kept as
  `flat-lminit-s{1,2}-crashed-fd`; relaunched fresh with `ulimit -n 65536`
  and `num_workers=6` on gruenau7:0 (s1) and gruenau10:0 (s2). Side effect:
  throughput improved 0.176 -> 0.10 s/problem with 6 workers.
  ACTION: audit every remaining pre-fix job.sh in this round for the same
  two settings.
  Their epoch-0 curve is still valid and reconfirms the depth penalty
  independently (s1, 100 episodes): feasible_menu .99 -> .99 (degenerate,
  random .98); full_catalogue .74 -> .46 (random .59); prior_propose
  .63 -> .46 (random .59), with imagined-step illegality .54 under
  full_catalogue at depth 4 -- i.e. over half the actions the model imagines
  taking at depth 4 are not legal, which is what the drift measurement and
  the rollout-trained cells are meant to fix.
- Root-caused and fixed at source (commit 12b1510, snapshot
  `runs/autonomy/_code/12b1510495050ca501cf8b111e61e923490f1782`):
  `scripts/train_flat_jepa.py` now calls
  `torch.multiprocessing.set_sharing_strategy("file_system")`. Dataloader
  workers previously passed every tensor by file descriptor, exhausting the
  process fd limit mid-run. `flat-lminit-ecf1-s0` died of exactly this
  (`RuntimeError: Too many open files`) and is relaunched from the new
  snapshot on gruenau7:1 (old dir kept as `-crashed-fd`).
  Caveat: the 8 cells already running still use the pre-fix snapshot and
  remain exposed until they finish; relaunch on crash rather than restart.

## 2026-08-20 — representation analysis: one strong result, one dead claim

Full report: `research/reports/intent_phrase/2026-08-20-representation-analysis/`
(commit cc8e678; mirrored to TextJEPA-paper, 129 figures). Every comparison
carries a **random-init control**, which is what makes the reading honest.

**Holds up — consequence geometry.** Two independent steps taken in either
order lead to the same situation; the state should encode that. Because the
flat encoder starts as a copy of the token LM, this is a before/after on the
same weights: AUC .711 after JEPA training vs .568 at LM-init, vs .568 for
the token LM (reproduces exactly, sanity check) and .516 for random-init.
JEPA training is what creates this. On ProofWriter, adding a single "not"
moves the JEPA state 21x further than a reordering that changes nothing;
the token LM is *anti*-consequence (AUC .170) — a meaning-preserving
reorder moves it further than a completely different action.
CAVEAT: one seed, one epoch-0 checkpoint, 72 cases; and the stylized JEPA
*fails* the same order-invariance test (.538, chance). Not yet a paper claim.

**Does NOT hold up — "JEPA states probe better than LM states."** This was
to be the paper's largest section. It is not true:
resolved .969 (JEPA) / .906 (token LM) / .923 (sentence LM) / **.912
(random-init JEPA)**. The trained margin over an untrained network is
+.057. Operator-from-displacement .995 looks impressive until the random
controls read .960/.972 — it is architectural, not learned. Likewise the
stylized paraphrase AUC of 1.000 is matched by random-init. And the
sentence LM separates negation *better* than the JEPA (.590 vs .208).
The step's value is not decodable by anyone (R^2 <= 0 everywhere).
JEPA resolvedness also decays with depth (.951 -> .602 by depth 5) while
both LMs hold at 1.000 — same decay found on 08-12, now replicated.

DECISION IMPLIED: the probe battery cannot be the paper's centrepiece as
planned. The defensible version of the analysis story is the narrower,
control-backed one: JEPA training reshapes the state so that *actions with
the same consequence land together and negations land apart*, measured as a
before/after on identical weights. Everything else needs the random control
printed next to it or it will not survive review.

NEXT: `nocfrank` vs `ecf16` on the same order-invariance test, to see
whether the counterfactual-ranking energy term is what drives .568 -> .711.
Both cells are training now.

- **Drift is now essentially gone during training** (cells RUNNING).
  Cosine between the k-step imagined state and the true state:

  | steps imagined ahead | 1 | 2 | 3 | 4 | 6 | 8 |
  |---|---|---|---|---|---|---|
  | before rollout training | .607 | .571 | | .470 | | |
  | roll124 @ step 1940 | .948 | .930 | | .910 | | |
  | roll1248 @ step 2960 | .946 | .932 | .927 | .923 | .911 | .891 |

  The 4-step imagination went .47 -> .91; even 8 steps holds .89, i.e. the
  decay with depth is flat rather than falling. `energy_cf_depth_acc` .97 in
  both. Cost: 0.075-0.119 s/problem, no penalty vs the reference cell.
  `flat-lminit-ecf16-roll1248-s0` gruenau9:0, `-roll124-s0` gruenau10:0;
  ~2.1-3.3 h/epoch, 10 epochs in 21-33 h.
  OPEN: this is the imagination being accurate, NOT yet planning improving.
  The epoch-0 watcher gives the first depth-1 vs depth-4 planning comparison;
  if depth 4 still loses to depth 1 with drift removed, the scorer is the
  remaining suspect and the oracle-scorer 2x2 decides it.

## 2026-08-20 — test-time compute baselines: self-consistency is the real bar

Round `runs/autonomy/intent_phrase/2026-08-19-lm-testtime-compute-v1/`
(commits 92a99e8, fe7bbb9, abd8090). 200 episodes, iGSM-med in-distribution,
under the headline free-generation protocol. Compute is counted on a shared
axis: backbone token positions (proportional to FLOPs), generated tokens,
and for the JEPA also predictor/energy forwards.

| token LM (lr 3e-4) | N=1 | N=4 | N=16 |
|---|---|---|---|
| plain | .810 | .815 | .815 |
| sample-and-rerank (sum logprob) | .810 | .650 | .525 |
| self-consistency on the answer | .795 | .880 | **.930** |
| ORACLE pass@N (upper bound) | .810 | .895 | .955 |
| token positions / episode | 29k | 112k | 442k |

lr 1e-3 ckpt: self-consistency .815 -> **.950**, pass@16 .970.

1. **Sample-and-rerank buys nothing and ranking by total log-probability
   actively hurts** (.81 -> .53): the model's own likelihood is not a
   verifier, and summed log-prob rewards stopping early.
2. **Self-consistency (sample 16 solutions, take the majority answer) is the
   real baseline, +.135.** Every depth claim we make must be stated against
   this, not against greedy decoding. Correcting the earlier framing.
3. pass@16 is .96 — a correct solution is almost always among the samples;
   the entire gap is *selecting* it. That is exactly the job we claim an
   energy head should do, and it is a good motivating number for the paper.
4. **The "looped LM" rows are NOT a looped LM.** No recurrent checkpoint has
   ever been trained; the finished LMs are fixed 12-layer. Feeding one a
   repeated prefix is off-distribution and destroys it (.81 -> .000). The
   architecture (`LoopedTransformerEncoder`, `model.recurrent=true`) exists;
   the checkpoint does not. A genuine looped baseline must be TRAINED.
   Do not report the current looped rows as a baseline.
5. JEPA on the same axis (`flat-lminit-s0/last.pt`, 50 eps): full_catalogue
   depth 1 **.94 at 9.6k token positions and zero generated tokens** — ~46x
   cheaper than self-consistency at 442k for a comparable score. But depth 4
   .52 and depth 16 .36, so the cheapness claim currently rests on depth 1.
   Interface-matched prior_propose depth 1 .82 at 168k token positions.

Sentence LM skipped: no current-recipe iGSM-med checkpoint yet (Alex).
NEXT: train a real recurrent token LM so the looped comparison exists.

## 2026-08-20 — depth diagnosed: the SEARCH FORMULATION is broken, not drift

Commits 8644dae, 0a6f850, e6ad7d1; snapshots archived under
`runs/autonomy/_code/`. All 2x2 rows below are CANDIDATE-PRIVILEGED ORACLE
DIAGNOSTICS — never paper rows.

**The 2x2 (ecf16 last.pt, 100 eps, prior_propose).** Cross the scorer
(learned energy vs a perfect goal metric) with the endpoints (imagined vs
truly executed):

| scorer x endpoints | d1 | d2 | d4 | d8 |
|---|---|---|---|---|
| A energy x imagined | .850 | .420 | .360 | .360 |
| B oracle-distance x imagined | .520 | .390 | .370 | .380 |
| C energy x TRUE executed | .830 | .410 | .370 | .390 |
| D oracle-distance x TRUE executed | .860 | .440 | .410 | .400 |

Row D is the full upper bound: perfect states AND a perfect goal metric.
**It collapses with depth exactly like row A.** So:
- A ~= C => **drift is NOT the cause.** Handing the planner the true
  executed states instead of imagined ones changes almost nothing.
- A > B => the learned energy is *better* than the oracle goal-distance it
  is trained to imitate (.85 vs .52 at depth 1). The energy head is not the
  bottleneck either.
- CORRECTION to the 08-19 entry: latent drift was the stated lead suspect
  and it is now ruled out as the cause of the depth collapse. The drift
  measurement and the rollout-trained cells remain valid and worth having
  (drift is real and severe, see below), but they will not by themselves
  make depth pay off.

**The actual mechanism, measured.**
1. Scoring only the ENDPOINT is blind to wasted steps. An illegal action is
   a no-op, so [illegal, a, b] reaches the same endpoint as [a, b] at no
   cost. From depth 2 the argmin is free to start with an illegal move:
   invalid rate of the *executed* action .02 -> .17 (d1->d4) in ALL FOUR
   rows including the full-oracle one.
2. Those illegal picks ended episodes. Each was masked "attempted", the
   proposer only emits ~1.6 unique parseable intents per state, and once all
   were masked the planner gave up: no-proposal episode rate .15 (d1) ->
   .58-.64 (d>=2). That is most of the prior_propose collapse.
3. Expansion was random, not the pool: a legal action was present in the
   pool at 98.8-100% of imagined depth-2/3/4 states, but the random tail
   draw picked a legal one only 38.8-45.6% of the time. Roots were being
   ranked by the luckiest of ~2-3 junk continuations.

**Drift IS severe on the reference (no-rollout) checkpoint** — 60 problems,
teacher-forced, LN space (the space the loss uses):

| k | 1 | 2 | 3 | 4 | 6 | 8 | two UNRELATED real states |
|---|---|---|---|---|---|---|---|
| cosine | .928 | .853 | .771 | .687 | .496 | .310 | .945 |
| LN relL2 | .370 | .535 | .670 | .785 | .998 | 1.169 | .326 |

Already at k=1 the imagined state is farther from its target than a random
unrelated real state is. No 2e6 explosion (norms inflate 25-80% with k).
Energy legality AUC decays much more gently (d0 .818, d4 .838, d8 .734) —
it reads legality off geometrically wrong states.

**Eval-time fixes applied** (depth 1 bit-identical in all cases; legacy
paths kept for ablation): energy-guided beam expansion instead of random
tails; `--aggregate mean_prefix` (default) charges every imagined step
rather than only the endpoint; when all candidates are masked, clear the
mask and retry instead of ending the episode.

| interface | depth | before | after |
|---|---|---|---|
| full_catalogue | 1 | .965 | .965 |
| full_catalogue | 2 | .865 (inv .44) | **.920** (inv .30) |
| prior_propose | 2 | .420 | **.575** |

(30-ep spot check of the full fix at prior_propose d2: .42 -> .67, imagined
invalid rate .376 -> .032.) The checkpoint has also trained further since
the 08-19 tables: full_catalogue d1 is now .965 (was .88), prior_propose d1
.850 (was .64).

**Honest status: depth still does not HELP, it just hurts less.** The 2x2
says the rest is not fixable at eval time — the objective rewards reaching a
goal-like endpoint and never the ordering or cost of the path.
NEXT (training change, not eval): make the energy score PARTIAL TRAJECTORIES
— rank counterfactual prefixes by whether the observed continuation followed,
at imagined horizons 2/3/4, using the existing rollout-counterfactual
machinery.

- **Real looped-LM baseline now training** (round
  `runs/autonomy/intent_phrase/2026-08-19-lm-looped-real-v1`, snapshot
  `_code/09f56579ffc60bd808deb8b1f09e89b64f881b4f`): `looped-med-lr1e3-s0`
  and `looped-med-lr3e4-s0` on gruenau12 GPUs 2/3 (L40), ~7-8 h train + 1-2 h
  eval. Recipe is byte-for-byte the 2026-08-18 med-scale LM recipe
  (`lm_loss_on=all_solution`, caps 15/20, 100k fresh problems/epoch x 30)
  except `model.recurrent=true` and bf16.
  Fairness for reviewers: fixed baseline 90.8M params, 12 block applications
  per forward; looped 12.8M params (one shared 7.09M block + 5.7M
  embed/pos/norm), K applications. ~7x fewer params, K/12 of baseline FLOPs;
  K=12 is equal block work and the eval axis K in {1,2,4,8,16} brackets it.
  Deliberately NOT parameter-matched by widening the block: that would move
  d_model off the frozen recipe and make K=8 cost ~8x baseline. Extra loops
  are then unambiguously test-time compute, not extra capacity. Training
  loop count is Poisson-lognormal (Geiping et al. 2025) mean 8 sigma 0.5
  clipped to [1,16], so the whole eval axis is in-distribution — the thing
  `looped_reread` lacked.
  NOTE: gruenau12 exists and had five genuinely idle L40s; add it to the
  survey list.

- **bf16 landed in `scripts/train_lm.py`** (commit 09f5657): `train.precision`
  was a dead key and is now live (unknown values error instead of being
  ignored); the same `set_sharing_strategy("file_system")` fd fix was needed
  here too and is applied. A/B at 3 epochs, identical seed, only precision
  differing: wall 246s -> 222s, val loss 3.334 vs 3.358, train curves track
  step-for-step from step 0. Safe, and in use.
  CORRECTION: the expected ~2x speedup is **~1.1x**. At batch 16 with short
  iGSM sequences these runs are dataloader/launch bound, not matmul bound.
  Do not claim 2x anywhere.

- **hard21 local token LM OOM'd and is relaunched.**
  `2026-08-19-lm-hard21-local-v1/tok-lm-hard21-fullsol-s0` died with
  `torch.OutOfMemoryError` on a 22GB card (faithful hard split needs
  `model.max_len=4096`). Old dir kept as `-oom`. Relaunched on gruenau12
  GPU 4 (L40, 46GB) from snapshot `_code/09f5657...` with `train.precision=bf16`
  and `train.num_workers=6`; steady-state ~18.9GB, comfortable headroom.
  This cell is what unblocks `flat-hard21-lminit-s0` (job.sh.pending).

- **Steps-distribution figure unblocked** (commit 043b34f). CORRECTION to the
  earlier note that reference-policy step counts were never recorded: they
  are — `_reference_episode` tracks steps for both the random and the
  first-feasible policy and both go through the same `summarize`, which
  already reports mean / median / p90 / steps-over-necessary. The real gap
  was narrower: only the planner's PER-EPISODE records were returned, so a
  distribution could be plotted for us but not for the baselines.
  `evaluate_flat_planning` now also returns `episodes_random` and
  `episodes_first_feasible`. Purely additive; existing keys unchanged.
  This is what the owner asked for: show our model solving problems in
  fewer steps as a distribution, not as a scored budget.

## 2026-08-20 — prefix-scoring energy launched (the depth fix)

Round `runs/autonomy/intent_phrase/2026-08-20-prefix-energy-v1/`, snapshot
`_code/c8115c9d23f35eca3c026cf88ff6c000a341e1ff` (includes the 12b1510
file_system fd fix). Commits 6c6a39a (objective) and c8115c9 (bug fix).

**Design.** The old energy answered only "how good is the place this rollout
ENDS UP?", which is blind to the path. New term `energy_prefix_rank` scores
whole paths: take the rollout the environment actually produced (a_1..a_H);
build a counterfactual by INSERTING one infeasible intent (from the existing
`rollout_counterfactual_k` env-driven negatives) at a random depth and
DROPPING the last true action, so both paths spend the same imagined budget
and the counterfactual simply wastes one step — exactly the degeneracy
endpoint scoring cannot see. The head emits an energy at every prefix; each
path is reduced by the mean over prefixes (matching the planner's
`--aggregate mean_prefix`; `endpoint` kept as the blind ablation) and
contrasted with the same `softplus(E_obs - E_cf)`.
CONTRACT CHECK: the only label is which continuation actually occurred,
which is in the data. No steps-to-go, no remaining-step counter, no symbolic
ranking label. Complies with the no-symbolic-heads rule.

**Bug caught mid-flight, worth remembering.** The first version APPENDED the
junk step, making every counterfactual path one step longer.
`energy_prefix_acc` hit 1.000 within 40 optimizer steps — the head had
learned to read PATH LENGTH, not the wasted step. At eval every candidate is
rolled to the same depth, so a length cue transfers nothing and the term
would have been silently inert while looking perfect. Length-matching fixed
it: accuracy back at chance at init (.496/.457) and climbing honestly
(.42 -> .72 over 80 steps). Three cells stopped and relaunched.
GENERAL LESSON: a term that saturates almost immediately is a shortcut
suspect, not a success.

**Smoke evidence.** `tests/test_energy_prefix_rank.py`, 7 tests passing, plus
from the real launched cell: depth-1 prefix energy equals the existing
one-step anchor energy to 1.19e-07 over 68 pairs (same root, same action,
same head — catches wrong index order or wrong s_0); all counterfactual
paths length-matched; grads non-zero (encoder 3.9e-3, predictor 1.99e-2,
energy head 8.29e-2); with the flag off the prefix loss is exactly 0.0 and
`energy_cf_feasibility_rank` is bit-identical (delta 0.00e+00); unread-key
guard aborts on a bogus key. Cost 0.089-0.092 s/problem — no slowdown.

| cell | host:GPU | recipe |
|---|---|---|
| `ecf16-prefix4-s0` | gruenau12:5 (L40) | ecf16 + energy_prefix_rank 4 |
| `ecf16-prefix4-roll124-s0` | gruenau12:6 (L40) | + latent_rollout_pred 1, ks [1,2,4] |

~2.8 h/epoch, 10 epochs ~28 h. Watcher every 2 epochs, 100 eps, NEW eval
defaults: full_catalogue at depths 1/4/8 and prior_propose at 1/4 — depth 1
measured alongside every deep number, since the claim is "depth HELPS".
Final on best.pt: 300 eps, depths 1/2/4/8/16 ID and 1/4/16 OOD.
A third weight arm (`prefix16`) is generated but NOT launched: the only free
GPUs were Turing-class where bf16 is ~8x slower (0.78 vs 0.10 s/problem, a
10-day run). Needs one free A100/A6000/L40.
- hard21 local token LM is healthy after the L40 relaunch: epoch 0
  val_loss 0.9096 in ~26 min, so ~13 h for 30 epochs. (My earlier "no step
  line" worry was a wrong grep — this script logs per-epoch, not per-step.)
- Third arm `ecf16-prefix16-s0` launched on gruenau10:2 (A100, 62GB free at
  launch; packed alongside an existing cell). Completes the dose-response on
  the new path-scoring term: weight 4 (x2, one with the rollout drift fix)
  and weight 16. A dose-response is far more defensible than a single
  working setting. Training normally at 0.19 s/problem, 14GB.
  All other GPUs genuinely busy; gruenau8's four A6000s still show 45GB
  allocated at 0% util (held-but-idle foreign job) and are left alone.

- **RETRACTED: the epoch-0 prefix-energy numbers (.770 / .780).** When the
  superseded length-cue cells were stopped, the trainer was killed but the
  watcher subshell was not. Its `train_finished` sentinel moved with the
  directory rename, so its loop never terminated, and it kept polling the
  ORIGINAL path string — which had since been recreated as the live cell. It
  wrote `watch/epoch0/plan_*.json` there using the OLD length-cue code, and
  the live watcher's "skip if the file exists" guard then skipped its own
  proper evaluation. Tell-tale signs: the depth-1 result appeared implausibly
  fast, and the eval log showed duplicated `[ep 10/100]` lines.
  Cleanup: four orphaned watcher shells and two evals (all from snapshot
  6c6a39a) killed; `watch/epoch0`, `curve.jsonl` and eval logs deleted on
  both live cells; incident written up in the round `NOTES.md`.
  TRAINING WAS NEVER AFFECTED — both cells ran from the correct snapshot
  c8115c9 throughout.
  LESSON (now in NOTES.md): when relaunching into a REUSED round directory,
  kill the watcher subshell explicitly, not just the trainer. A stale watcher
  writing into a recreated path is silent and looks like a real result.
  Epoch 0 discarded; first trustworthy depth-1 vs depth-4 comparison is the
  epoch-2 pass, ~3-6 h out.
- Training signal meanwhile is strong: `energy_prefix_acc` .966 / .954 (from
  chance at init) — the head reliably separates a path that wasted one
  imagined step from the path that actually occurred, which is exactly the
  discrimination the endpoint-only objective never had.

- **Ablation: the counterfactual-ranking term does NOT cause the consequence
  geometry.** Ran `scripts/analysis/consequence_geometry.py` on faithful,
  120 problems / 72 pairs, over three cells that differ only in
  `energy_cf_feasibility_rank.weight`, at comparable training (steps 10000 /
  10500 / 11000):

  | weight | state order-invariance AUC (cosine) | AUC (L2) | Cohen's d |
  |---|---|---|---|
  | 0 (`nocfrank`) | **.774** | .765 | 1.011 |
  | 16 (`ecf16`) | .701 | .705 | .716 |
  | 64 (`ecf64`) | .698 | .692 | .764 |

  Not a dose-response — the opposite, mildly. And the arm with the MOST
  steps (ecf64, 11000) scores lowest, so this is not a training-amount
  artifact. Checkpoints were copied before reading, since the cells are
  live and writing `last.pt`.
  READING: all three remain well above the LM-init baseline (.568), so JEPA
  training as a whole still causes the effect — but it is NOT the energy
  counterfactual-ranking term that does it. The natural next suspects are
  the terms shared by all three arms (`latent_pred`, `counterfactual_state`,
  `observed_action_ldad`, vicreg). The paper must not attribute the geometry
  result to the energy objective.
  CAVEAT: one seed, epoch-1 checkpoints, 72 pairs. Worth re-running at
  convergence before it goes in the paper.

- **Duplicate-run incident: `flat-lminit-frozen-s0` was running TWICE**, on
  gruenau7:1 (the host recorded in the round's HOSTS.txt) and an
  unregistered duplicate on gruenau2, both writing the SAME `model/` dir.
  The gruenau7 copy had already crashed in training with the file-descriptor
  bug (exit 1, but its `state` file still read RUNNING because job.sh only
  writes the final state after the eval stage) and moved on to final eval;
  the gruenau2 duplicate kept training and overwrote `last.pt` and
  `metrics.csv` underneath it.
  Resolution: killed the gruenau2 duplicate by PID (28657/28661/28662; never
  `pkill -f`). `best.pt` is dated 16:40, BEFORE the duplicate started
  (~16:46), so the final eval reading it is clean. `last.pt` (20:59) is
  contaminated and was dropped from the geometry ablation.
  HOW IT WAS CAUGHT, worth reusing: the checkpoint reported step 1500 while
  the training log was at step 11280. Checkpoint-step vs log-step mismatch is
  a cheap integrity check — apply it before trusting any checkpoint.
  TWO REAL BUGS EXPOSED: (1) `state` can read RUNNING for a cell whose
  training has already failed, so `state` alone is not a health check —
  always check `exit_code` and the log tail; (2) nothing prevents two
  launches of the same cell into one run dir. A lock file or a
  refuse-if-state-exists guard in job.sh would prevent recurrence.

- **Term isolation: what actually builds the consequence geometry.** Same
  test as above (faithful, 120 problems, 72 pairs), four arms that differ
  only in one ingredient. Steps are noted because they are NOT matched:

  | arm | steps | AUC (cos) | AUC (L2) | Cohen's d |
  |---|---|---|---|---|
  | full recipe | 13500 | .797 | .795 | 1.089 |
  | minus energy ranking | 10500 | .774 | .765 | 1.011 |
  | **minus intent_prior_lm** | 8500 | **.938** | .933 | 1.905 |
  | **from scratch (no LM init)** | 13000 | **.879** | .880 | 1.269 |
  | (reference: LM-init before any JEPA training) | 0 | .568 | | |

  TWO FINDINGS, both against the current framing:
  1. **The generative `intent_prior_lm` term is the main thing SUPPRESSING
     the geometry** (.938 without it vs .797 with, on FEWER steps — so this
     is not a training-amount artifact, the direction is against the
     confound). Intuitive: a term that forces the state to keep enough
     surface detail to regenerate the exact tokens works directly against
     collapsing paraphrases onto one point.
  2. **LM initialization HURTS the geometry**: from scratch .879 beats
     LM-init .797 at comparable steps (13000 vs 13500). Consistent with the
     08-20 finding that the token LM's own geometry is anti-consequence
     (ProofWriter AUC .170). Starting from it is a handicap, not a head
     start.
  REFRAMING NEEDED: the "before/after on identical weights" story
  (.568 -> .711/.797) is still true and still the cleanest CAUSAL evidence
  that JEPA training builds this structure, but it must NOT be presented as
  if LM init were beneficial. The stronger claim available is that JEPA
  training builds consequence geometry from scratch, and does so BETTER
  without the generative prior.
  REAL TRADE-OFF TO QUANTIFY BEFORE ACTING: `intent_prior_lm` is also the
  anti-collusion anchor and the source of no-menu proposals
  (`prior_propose`). Dropping it would buy geometry and lose the menu-free
  interface. Needs the planning numbers from `flat-lminit-noprior-s0` side
  by side with the geometry before any recipe change.
  CAVEATS: one seed, 72 pairs, mid-training checkpoints at unequal steps.
  Re-run at convergence with matched steps before this goes in the paper.
- The `intent_prior_lm` trade-off is MILDER than first stated (correcting my
  own note above). At the only fairly matched point available (epoch 0 for
  both), dropping the term costs essentially nothing in planning:

  | | full recipe | minus intent_prior_lm |
  |---|---|---|
  | full_catalogue d1 / d4 | .720 / .480 | .720 / .490 |
  | prior_propose d1 / d4 | .560 / .510 | .540 / .480 |
  | proposal recall | .912 | .894 |

  So the menu-free interface does NOT collapse without the term — the
  proposal head is initialized from the token LM and keeps producing usable
  intents (recall .894) even with the loss weight at 0. My earlier claim
  that dropping it "loses the menu-free interface" was wrong.
  STILL UNRESOLVED: epoch 0 is early, and the term's value may only appear
  as training matures. Need both arms at the SAME later epoch before acting.
  (The epoch-2 comparison I first pulled was confounded — noprior was at
  epoch 0 and the full recipe at epoch 2.)

- **Integrity sweep of all 18 running cells** (prompted by the frozen-cell
  duplicate). Found one more silently-dead run: `flat-lminit-nocfrank-s0`
  had crashed with the same file-descriptor bug at step 10680 roughly five
  hours earlier, but showed as RUNNING the whole time — job.sh only writes
  its final state AFTER the eval stage, and the eval stage was spinning in
  `while [ ! -f train_finished ]`, which could never become true. Killed the
  three stuck shells by PID on gruenau7, marked FAILED, relaunched on
  gruenau7:2 from the fd-fixed snapshot 12b1510 (confirmed training,
  `energy_cf_feasibility_rank=0.0000` as the arm requires; old dir kept as
  `-crashed-fd`).
  Its step-10500 checkpoint PREDATES the crash, so the geometry ablation
  that used it stands.
  Rest of the sweep: two cells carry a stale `exit_code` from an earlier
  crash-and-relaunch today (benign, both live and advancing), the frozen
  cell is legitimately in final eval, everything else healthy.
  STANDING CHECK, use it routinely: `state` is NOT a health check. Read
  `state` + `exit_code` + whether the log step is still advancing. A cell
  whose log step has not moved in hours is dead regardless of what `state`
  says.

- **The 08-20 eval-time fixes made DEEP search enormously more expensive —
  a real cost the entry above does not mention.** Under the new defaults
  (energy-guided beam expansion + `mean_prefix` aggregation), a depth-8
  100-episode evaluation managed 10 episodes in 1h48m, i.e. ~18 h per run.
  Because the watcher runs depths in sequence, every later pass was queued
  behind it: the epoch-2 comparison would never have arrived and the
  epoch-0 backfill would never have fired. Silent, not an error.
  Fix: killed both depth-8 evals; placed sentinel JSONs at
  `watch/epoch{0,2,4,6,8,10}/plan_full_catalogue_d8.json` so the existing
  "skip if present" guard bypasses depth 8; same for depths 8/16 in
  `final_id` and depth 16 in `final_ood` (300 episodes there — far worse).
  Watchers now report full_catalogue d1/d4 and prior_propose d1/d4, which
  still answers the round's question because depth 1 sits beside depth 4.
  CONSEQUENCE FOR THE PAPER: any depth-8 or depth-16 number needs its OWN
  dedicated long-running cell; it cannot ride along in a training watcher.
  Budget for that explicitly when planning the final runs.
  Training unaffected: epoch 0 complete on both cells, into epoch 1 at
  ~1.44 s/step; the epoch-0 backfill now runs detached on the cluster so it
  does not depend on any agent's background task surviving.

- **hard21 token LM is working on the HARD faithful split.** Strict free
  generation (model writes the whole solution, no menu, no partial credit),
  50 episodes every 2 epochs:

  | epoch | 1 | 3 | 5 | 7 |
  |---|---|---|---|---|
  | success | .00 | .02 | .22 | .36 |
  | invalid-step rate | .82 | .65 | .29 | .17 |
  | unparseable-step rate | .23 | .11 | .053 | .011 |

  At epoch 9 of 30 and still climbing (val_loss .91 -> .27). Same
  `lm_loss_on=all_solution` fix that took iGSM-med from 0 to .82.
  For the steps figure: `solved_exact_necessary_frac` = .944 at epoch 7 —
  when it solves, it uses EXACTLY the minimum number of steps 94% of the
  time. The model is not wandering to the answer.

- **Watch item: `energy_prefix_acc` is at .986-.992 on all three prefix
  cells by epoch 1.** Not the instant saturation that exposed the earlier
  length-cue bug (it took ~an epoch, and started at chance), so the term is
  learning something real. But near-ceiling accuracy means the DISCRIMINATION
  MAY BE TOO EASY: if detecting one inserted infeasible intent is close to
  trivial, the gradient dies long before the head is shaped into a scorer
  that helps search. If epoch-2/4 planning shows depth still not helping
  while this sits at .99, the fix is a HARDER negative, not more weight —
  e.g. insert an action that is feasible but useless (a legal step that does
  not advance the solution) rather than an infeasible one, or insert deeper
  into the path where the consequences are subtler.
  Epoch-0 rows (step-500 sanity checkpoint, NOT evidence): prior_propose
  d1/d4 = .62/.40, .61/.46, .60/.45; full_catalogue d1 .76 where backfilled.
  Depth still hurts at this stage, as expected this early.

- **Epoch-2 depth-1 numbers (first non-sanity row):** `ecf16-prefix4-s0`
  full_catalogue d1 **.980** (energy AUC .949/.984); `ecf16-prefix4-roll124-s0`
  d1 **1.000** (.921/.982). The mature ecf16 baseline reaches .965 at d1, so
  the prefix term costs nothing at depth 1 and the rollout arm has already
  saturated it. Depth 4 still running — THAT is the number that decides the
  question. Epoch-0 sanity row had prior_propose .62->.40 and .61->.46.

- **Harder negatives built (commit 9d5dc3b), NOT launched** — no genuinely
  free GPU (the `gruenau-gpus` helper reported gruenau11:3 FREE but direct
  inspection showed 13.5GB held at 1% util by a foreign job; only Turing
  RTX 6000s are actually free and bf16 is ~8x slower there).
  **CORRECTION to my own proposal.** I suggested inserting an intent that is
  feasible but not the one taken. That is NOT a clean label here: rollout
  continuations are drawn UNIFORMLY AT RANDOM from `feasible_actions()`, so
  at depth >= 1 a feasible-not-taken action is statistically indistinguishable
  from the taken one — neither is better, the sampler just picked one.
  Training to rank the taken one lower would be fitting a coin flip and would
  drive the term back to 50%. Not built, correctly.
  What was built instead: `rollout_counterfactual_k` already draws half its
  negatives from ALREADY-RESOLVED variables — not literally legal (resolved
  variables are excluded from `feasible_actions`, so they no-op too) but
  *legal-looking*: the intent re-derives a fact the state already has, so it
  reads as a sensible sentence. Labelled "legal-looking but pointless", not
  "legal but useless", to avoid overclaiming.
  THE PRINCIPLED ROUTE, noted for later: make the rollout follow the
  REMAINING TRUE SOLUTION TRACE instead of random feasible actions. Then the
  continuation that occurred is the actual solution and feasible alternatives
  at each depth are genuinely off-path — the depth-0 anchor contrast extended
  to depth >= 1. That is a rollout-policy change in the dataset, not a
  sampler tweak.
  Switches, both defaulting to current behaviour: `cf_kind: all|premature|
  resolved` and `depth_bias: uniform|late` (mean insertion depth 2.079 ->
  2.349). Smoke: all five variants start at chance (.403-.427), all
  length-matched (asserted), grads reach encoder/predictor/energy head, and
  the dataset change is purely additive — every pre-existing collated tensor
  is bit-identical to snapshot c8115c9 on the same seed.
