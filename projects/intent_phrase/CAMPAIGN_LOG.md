# Intent-phrase campaign log (living handoff)

_Keep short. Compress completed stages into a few lines; details live in
`research/reports/intent_phrase/<date>-*/REPORT.md` and are mirrored to
`/vol/home-vol2/ml/laitenbf/TextJEPA-paper/reports/`. Last update: 2026-08-20._

## CURRENT STATE (2026-08-20, end of night) — READ THIS FIRST

A complete pick-up-from-here summary. Detail for every line is in the dated
entries below and in `research/reports/intent_phrase/2026-08-20-*/`.

### The one-sentence status
Both LM baselines now work (iGSM-med .82, faithful-hard .90 free generation).
The JEPA planner beats every test-time-compute baseline at depth 1 and at a
fraction of the compute. **Depth still does not help, and we now know why it
probably cannot in iGSM** (see "THE DEPTH ANSWER"). The representation
claims from before today have been withdrawn or narrowed after proper
controls.

### THE DEPTH ANSWER (most important insight of the day)
Depth helps when the depth-1 scorer is WEAK and hurts when it is STRONG:

| protocol | d1 | d2 | d4 |
|---|---|---|---|
| 2026-08 stylized, latent oracle-distance | .106 | .456 | **.534** (monotone UP) |
| 2026-08 stylized, exact symbolic distance | 1.000 | .534 | **.144** (monotone DOWN) |
| today, full_catalogue @cap1.0 | .725 | — | .310 |
| today, prior_propose @cap1.0 | .790 | — | .415 |

The old monotone-increasing curves came from a scorer so bad at depth 1
(.106) that extra search could still recover ground. That is not the
demonstration we want. Today depth 1 is genuinely good, and deeper search
makes the model pick a WORSE FIRST ACTION: executed invalid rate 1.3% -> 35%
(full_catalogue) and 1.7% -> 14.5% (prior_propose) going d1 -> d4.
**Structural reason: iGSM HAS NO DEAD ENDS.** Every legal action is either
necessary or merely wasteful; nothing forecloses a later solution. So
lookahead has almost nothing to discover, while it does add imagination
error. Verified: this is not a budget artefact — depth 1 wins at every cap
from 1.0 to 4.0.
**TO GET MONOTONE DEPTH GAINS we need choices that FORECLOSE** —
irreversibility, dead ends, consumable resources. Blocksworld has exactly
this (un-stacking is required). This is a paper-level finding, not a failure.
Contributing but secondary: the energy head is a strong LEGALITY detector at
every depth (.95-.99) but a weak GOODNESS ranker that collapses off the root
(progress AUC .842 d0 -> .695 d1 -> .615 d2 -> .606 d4).

### DECISIONS IN FORCE (owner)
1. **Success = the model EMITS THE ANSWER explicitly** (`gen_outcome=model`,
   `success_answer_rate`) for BOTH families. Only definition under which the
   planner and the LM do the same task; random is 0 at any budget. NOT yet
   applied to the running watchers — this is the top pending change.
2. **Report the budget curve** (cap 1.0 / 1.25 / 4.0), never a single cap.
   `scripts/rescore_budget.py` re-scores any completed plan JSON for free.
3. **Traps should scale with problem size** — distractors as a fraction of
   legal actions, so bigger problems get proportionally MORE traps. Not yet
   implemented.
4. **Target architecture**: learned prior over action codes (codebook
   preferred, flow fallback) + rollout ENTIRELY IN LATENT SPACE + energy
   selection + a DETACHED decoder rendering to text only at execution. Model
   never sees a menu, list or feasibility signal.
5. **Step-efficiency is ONE result, not the lens on every experiment.**
   Success rate is primary.
6. No symbolic or hardcoded heads (CLAUDE.md). Symbolic state is
   evaluation-only and must be labelled.

### WHAT IS RUNNING (all verified 2026-08-20 end of night)
Round `2026-08-19-flat-jepa-v1` (iGSM-med, 13 cells): `flat-lminit-s0/s1/s2`
(seeds), `flat-scratch-s0` (no LM init), `flat-lminit-frozen-s0`,
`flat-lminit-lr3e5-s0`, dose-response `nocfrank`/`ecf1`/`ecf16`/`ecf64`,
`flat-lminit-noprior-s0`, `flat-lminit-ecf16-roll124-s0` and `-roll1248-s0`
(multi-step rollout prediction).
Round `2026-08-20-prefix-energy-v1` (4 cells): `ecf16-prefix4-s0`,
`-roll124-s0`, `-prefix16-s0`, `-premature-s0` (the hard-negative arm).
Round `2026-08-20-detach-lm-state-v1` (2 cells): `ecf16-prefix4-detachlm-s0`
and matched control `ecf16-prefix4-tied-s0`.
Round `2026-08-19-flat-jepa-stylized-v1`: `flat-stylized-s0`.
Round `2026-08-19-flat-jepa-hard21-v1`: `flat-hard21-scratch-s0` (relaunched
on gruenau9:1 after an unexplained exit 1 at ep2 step 13620 — stderr showed
only a warning, suspect an external kill; WATCH IT). `flat-hard21-lminit-s0`
is ARMED with the finished hard21 LM checkpoint but NOT YET LAUNCHED.
Round `2026-08-19-lm-hard21-local-v1`: token LM in final 200-episode eval.
Round `2026-08-19-lm-looped-real-v1`: COMPLETE, both seeds.
Alex `2026-08-19-alex-lm-finals-v1`: 7 LM cells, all still PENDING on
priority, not blocking anything.
IN-FLIGHT AGENT ROUNDS with no cells yet: true-oracle upper bound;
solution-following rollout policy; latent planning + detached decoder.

### HEADLINE NUMBERS (all with their caveats)
- Token LM free generation: iGSM-med **.82**, faithful-hard **.90** (0% before
  the `lm_loss_on=all_solution` fix). Solves in exactly the minimum number of
  steps 94% of the time.
- Test-time compute, one axis (token positions/episode): greedy .81 @29k;
  sample-16 + confidence rerank **.53** @442k (HURTS); sample-16 + majority
  **.93** @442k; looped LM 8x **.73** @217k (saturates, 16x gains nothing);
  **JEPA depth 1 .94 @9.6k**. pass@16 is .96 — the whole gap is SELECTION.
- Planning, budget curve, full_catalogue ID d1: cap1.0 **.740** vs random
  .000; cap1.25 .855 vs .010; cap4.0 .995 vs .595. **Our margin over guessing
  was understated ~2x by the generous budget.**
- Codebook proposer: **.027 -> .890** on a current checkpoint, purely from
  today's fixes. BUT it is CATALOGUE-PRIVILEGED (snaps to the problem's own
  action list); truly catalogue-free scores exactly **.000**.
- `prior_propose` IS a genuine no-menu-at-all interface (writes phrases token
  by token, exact-match grounding only): **.620 ID @cap1.25 vs random .010**,
  and a progress-making action is in its top 4 for **96%** of states.

### WITHDRAWN / CORRECTED CLAIMS (do not resurrect)
- "JEPA states probe better than LM states" — FALSE with proper floors.
  Set-valued state readout is AT FLOOR for every arm in both domains (a step
  counter beats every encoder). Only ~+.07 of the binary-probe margin is
  training. The v1 probe table is WITHDRAWN.
- "Oracle-goal planning scored 1.0 before" — protocol conflation. It never
  exceeded .756 (privileged) / .21 (honest). The 1.0 rows were ExpertReplay
  and exact symbolic distance. Two reports carrying this error have been
  corrected in the repo AND the paper mirror.
- "`--scorer oracle_distance` is an upper bound" — it is NOT. It encodes the
  true solved state with our own model and ranks by latent L2, so it measures
  the REPRESENTATION. The real symbolic upper bound is being measured now.
- "full_catalogue / feasible_menu are meaningful comparisons" — both are
  SATURATED REFERENCE COLUMNS at cap 4 (random .595 / .975).
- "OOD is harder" — FALSE as configured: useless-action fraction falls
  .407 -> .181 because bigger problems use a larger share of their catalogue.
- "The energy counterfactual-ranking term drives the consequence geometry" —
  FALSE, the ablation without it scores HIGHER (.774 vs .701).
- "bf16 gives ~2x" — it gives ~1.1x; these runs are dataloader-bound.
- "Latent drift causes the depth collapse" — ruled out by the oracle 2x2.
  Drift is real and severe (fixed by `latent_rollout_pred`) but not the cause.

### OPERATIONAL TRAPS (all cost us time today)
- **Dataloader fd exhaustion killed SIX cells.** Fixed at source
  (`set_sharing_strategy("file_system")`, commit 12b1510) but any job.sh
  built from an older snapshot is still exposed. ALWAYS `ulimit -n 65536` as
  line 2 and `train.num_workers=6`.
- **`state` is NOT a health check.** job.sh writes the final state only AFTER
  the eval stage, so a crashed cell can read RUNNING for hours. Check
  `state` + `exit_code` + whether the log step is still advancing.
- **Nothing prevents launching the same cell twice** into one run dir. It
  happened (frozen-s0, two hosts, same model/ dir). Detect via
  checkpoint-step vs log-step mismatch.
- **Kill the WATCHER subshell explicitly** when relaunching into a reused
  round dir; a stale watcher writes results from old code into the recreated
  path and looks like a real result.
- **Do not delete live eval output**; `plan_flat.py` opens its `--out` up
  front, so files deleted mid-pass vanish silently.
- **`configs/flat_jepa_hard21.yaml` and `flat_jepa_stylized.yaml` are
  STANDALONE** — they do not inherit from `flat_jepa.yaml`. Every new
  objective key must be added to all three or those runs abort with KeyError.
- **A term that saturates almost immediately is a shortcut suspect.** The
  first prefix-energy version hit 1.000 in 40 steps by reading PATH LENGTH.
- **Turing cards (Quadro RTX 6000)**: bf16 falls back to fp32 and OOMs; use
  `train.microbatch_size=4`. They are also ~8x slower.
- **gruenau8's four A6000s** show 45GB at 0% util — held-but-idle foreign
  job, leave alone. **gruenau12 exists** (ten L40s) and was missing from the
  survey list for most of the campaign.
- Never `pkill -f` on gruenau1 (kills the shell). Kill only own PIDs.
- Copy `last.pt` before reading it — cells are live and writing.

### TOP PENDING WORK, in priority order
1. Apply the answer-emission success criterion to every watcher and re-run
   the headline tables under it.
2. Land the solution-following rollout policy (training rollouts currently
   pick uniformly at random among legal actions, so "what happened next"
   carries no quality information — the likeliest cause of the goodness
   collapse off the root).
3. The true symbolic-oracle upper bound; if it is ~1.0 the loss decomposes
   cleanly into representation + energy.
4. The decoder gate for the target architecture: can a context-conditioned
   detached decoder exactly reconstruct HELD-OUT action phrases? If not, the
   latent-planning architecture is not viable and we must know early.
5. Implement traps-as-a-fraction, and re-establish a genuinely harder OOD
   band.
6. Launch `flat-hard21-lminit-s0` (armed, not launched).
7. Re-run the geometry and probe comparisons at MATCHED training steps on
   converged checkpoints before anything goes in the paper.


### 2026-08-21 consolidation checkpoint (main session)
- ANSWER-EMISSION CRITERION LANDED (commit e1c1f93): model must generate the final outcome sentence itself and terminate by choice; planner .720/.830 (cap 1.0/1.25) vs random .000/.010; random at cap 4 halves to .285; LM re-scored .815 hard / .795 med. Report: research/reports/intent_phrase/2026-08-21-answer-emission/. One cell still running: ood_prior_d1 (gruenau1 GPU2); rescore with scripts/rescore_budget.py when plan.json lands.
- flat-hard21-lminit-s0 LAUNCHED on gruenau9 GPU2, warm-start verified (latent_pred 0.22 at step 0 vs ~2+ scratch).
- All five subagents told to wind down and write HANDOFF.md in their round dirs (answer-emission-v1, imagined-energy-v1, rollout-solution-v1, true-oracle-upper-bound, action-decoder-gate-v1); GPU cells left running. Check those HANDOFF.md files first when resuming.


### 2026-08-21 agent wind-down: all five delivered (main session)
- ANSWER-EMISSION OOD landed: prior_propose d1 OOD .835 cap1.0 (random .005), identical under answer criterion. All four cells done; round complete.
- TRUE-ORACLE DECOMPOSITION (2026-08-20-true-oracle-upper-bound, commit 90cc2ac): symbolic oracle x true endpoints = 1.000 exact-minimum at ALL depths/caps -> search machinery fully exonerated; depth collapse is 100% the learned ruler. prior_propose ceiling .945 d1 / .920 d2, EVERY unsolved episode is proposal exhaustion (0 budget/invalid/bad-pick losses). Lat/energy-ruler ladders + beam-width controls still running on gruenau2 (RUNNING_PIDS.txt, summarize.py ready). CANDIDATE-PRIVILEGED, EVAL-ONLY labels everywhere.
- DECODER GATE PASSED (2026-08-20-action-decoder-gate-v1): held-out action-phrase reconstruction .9978 exact (.9992 on never-seen phrases). Ablations: action-only .28 novel, context-only .37, full .999 -> the old catalogue-free .000-parse obstacle is dissolved. Codebook prior trained (K=64 top-4 recall .764; use ep8 artifacts, 60-ep overfits). New planner interface code_prior wired (prior->dequantize->detached decode->exact-text ground), tests pass. NEXT: proposal bench vs token_head .959, then d1/d4 ID/OOD planning. RISK: decoder saw only on-path states; if mid-episode parse sags, retrain on randomized feasible trajectories.
- ROLLOUT-LABEL FIX (2026-08-20-rollout-solution-v1, in 5d1e8b8): training rollouts stepped uniformly at random among legal actions -> "observed continuation" at imagined depth carried zero quality signal (coin flip); explains legality-good/goodness-collapsed heads. New rollout_solution_prob (p=0 bit-identical). solp100-s0/solp050-s0 fine-tunes on gruenau11 + step-matched probe (matched_step11k/). Report all p values, never p=1.0 alone. TRAP: probe_flat_energy_progress.py was untracked -> git-archive snapshots omit it; commit it properly.
- IMAGINED-ENERGY FIX (2026-08-21-imagined-energy-v1, commits 03bb351/0706638): re-encoded off-reference states are at CHANCE (AUC .49 d4) -> collapse is head/encoder train-test gap, not imagination error. energy_imagined_rank objective; fine-tune imagined-ecf16-ft-s0 on gruenau11:3 (runs own final evals).
- tok-lm-hard21-fullsol-s0 hit 250000s TIMEOUT during a late free-gen eval; training + 3 main eval JSONs complete; only the last eval truncated.
- All round dirs have HANDOFF.md with PIDs and exact continuation commands.


### 2026-08-21 code_prior first end-to-end numbers (main session)
- Bench (roll124 ckpt, 611 states): code_prior K=16 recall_true_next .586 / recall_feasible .820 / parse .667 / no-proposal .003 vs token_head .961. Prior code-ranking is the bottleneck, not breadth or decoder.
- Planning cap1.0 d1: code_prior .115 (OOD .000, d4 .030) vs prior_propose .840 on SAME ckpt (best prior_propose yet; answer criterion within .01). codebook_ground collapsed to .010 on roll124 (was .890 on ecf16) -> codebook quality is strongly checkpoint-dependent.
- Wide-K test launched (K=32/64 proposals + --prior-samples 64, gruenau1): does breadth + energy selection recover recall? Results: runs/autonomy/intent_phrase/2026-08-21-code-prior-v1/eval-wideK/.


### 2026-08-21 wide-K + ecf16 retrain (main session)
- Wide-K (K=64 + prior-samples 64): true_next recall .715, planning d1 cap1.0 .215 (vs .115 at K=16) -> breadth helps but saturates; prior ranking is the ceiling.
- Launched ecf16-retrain (gruenau1:0): decoder + K=64 prior retrained on ecf16 ckpt (codebook_ground .890 geometry) then bench + d1 planning; results in runs/autonomy/intent_phrase/2026-08-21-code-prior-v1/ecf16-retrain/.


### 2026-08-21 codebook prior: geometry refuted, capacity sweep (main session)
- ecf16 retrain: recall .717 K=64 (same as roll124 .715), planning d1 cap1.0 .215 / cap1.25 .345 -> checkpoint geometry is NOT the lever; the prior model itself is (val_top1 .378, overfits). Fronts: architecture PROVEN (.998 gate), prior = the open research gap.
- Prior capacity sweep launched on ecf16 pair cache (K 32/64/128, hidden 512/1024, layers 2/3, ep 8/16): runs/autonomy/intent_phrase/2026-08-21-code-prior-v1/prior-sweep/.
- Flagship flat-lminit-ecf16-s0 DIED ep3 step 22120 of the PRE-FIX fd bug (launched from pre-fix snapshot); best/last ckpts + most final evals intact; CONTINUATION flat-lminit-ecf16-cont-s0 running on gruenau7:0 from last.pt on fixed code (also the longer-training test).


### 2026-08-21 codebook ladder complete; K-axis exhausted (main session)
- Ladder (all cap1.0 d1, 200 eps, answer-criterion equal): K=16 .115 -> wide-K64 .215 (cap1.25 .345, BEST) -> ecf16-geometry retrain .215 (geometry refuted) -> capacity/epochs flat (~.72 top4) -> K=32 coarse codes top4 .830 BUT decode ambiguity drops true_next to .555 and planning to .080. K-axis is a predictability-vs-ambiguity trade-off with no good point.
- DIAGNOSIS: MLP prior on pooled state vs token head reading full context (.96 recall). Same gap the decoder ablation exposed (context carries the names). NEXT: context-conditioned code prior (cross-attention over prompt tokens), agent launched.


### 2026-08-21 imagined-energy finished; combo launched (main session)
- imagined-ecf16-ft-s0 COMPLETED: d1 NEW BESTS full .820 / prior .845 cap1.0 (.900/.890 cap1.25) but depth still non-monotone (d2 .525/.570, d4 .285/.440). After-probe: progress AUC .809/.667/.653/.650 vs baseline .84/.70/.62/.61 -> decay flattened, level still ~.65. Cause: its imagined rollouts still stepped RANDOM-feasible, so the term mostly re-learned legality (its own acc .997 notwithstanding).
- COMBO cell combo-imgsol-s0 launched (gruenau11:3): energy_imagined_rank w8 + rollout_solution_prob 0.5, warm-start ecf16 — the two depth fixes are complementary; this is the decisive depth bet. solp050/solp100 fine-tunes at ep9+ on same host.
- K=32 planning .080 (worse): codebook K-axis definitively closed.


### 2026-08-21 context prior + off-path round (main session)
- CONTEXT-CONDITIONED PRIOR (commit 001c3ca) delivered the biggest codebook jump: top4 .720->.850, true_next .717->.766, planning d1 .285 cap1.0 / .395 cap1.25 (ladder .115->.215->.285). Still short of token head .840.
- Biggest remaining leak: 39% of decoded proposals fail to parse (decoder trained only on-path). New --offpath-prob in train_action_decoder (commit after 001c3ca): cache walks step onto random feasible actions with prob p; self-supervised.
- offpath-v1 chain launched (gruenau12:4): offpath 0.3 cache -> retrain decoder + ctx prior -> bench -> d1 plan. Results: runs/autonomy/intent_phrase/2026-08-21-code-prior-v1/offpath-v1/.


### 2026-08-21 documentation pass (main session)
- Three dated reports written and mirrored to the paper repo: research/reports/intent_phrase/2026-08-21-true-oracle-ladder/ (full ladder incl. the four-ruler decomposition table and beam-width control; oracle round 30/31 cells done, lat-true d4/d8 rows partial n=25-60 and labeled), 2026-08-21-imagined-energy/ (root causes A+B, fix-A-alone results, combo decision rule), 2026-08-21-codebook-autonomous/ (gate ablation, six-step planning ladder, coverage decision frame).
- offpath-v1 status: decoder gate on off-path cache PASSED .997 exact/.995 novel; ctx-prior retrain hit two infra bugs, both fixed+committed (disable fused MHA fast path in train_code_prior.py -- illegal memory access on context batches; batch 16 vs OOM from 4x longer off-path histories); attempt 3 running on gruenau12:4 with bench+plan chained.
- flat-lminit-frozen-s0 (frozen-encoder ablation) DIED ep1 step 11280 of the pre-fix fd bug; ckpts intact; not yet relaunched (lower priority; needs a free ~22GB GPU; continuation recipe = same as ecf16-cont with encoder_mode frozen).
- oracle ladder extra finding: even sym-true prior_propose declines .945->.830 with depth (longer committed prefixes between re-plans amplify proposal-dead states) -- proposer effect, not search.


### 2026-08-21 offpath result + union round (main session)
- offpath-v1 COMPLETED: autonomous codebook planning .340 cap1.0 / .480 cap1.25 / .565 cap1.5 (ladder .115->.215->.285->.340); proposal feas recall .961, true_next .779 at K=64. Off-path decoder held its gate (.997).
- union-prior launched (gruenau12:4): ctx prior trained on UNION of on-path+off-path caches (50589 pairs, off-path val), larger ctx d512/l3, 8 epochs, with the off-path decoder; then bench + d1 plan. Target: recover on-path top4 while keeping off-path gains.

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
- OPERATIONAL NOTE: on Turing-class cards (Quadro RTX 6000, 24GB) bf16
  autocast falls back to fp32, so the flat-JEPA recipe (~19GB on an L40)
  OOMs there. Workaround: `train.microbatch_size=4` with gradient
  accumulation to the same effective batch of 16 -> 17.2GB. Relevant because
  the RTX 6000s are the cards most often actually free, and this is part of
  why they are ~8x slower for us.

- **The "resolved" negative is NOT harder — it is easier. Hypothesis dead.**
  120-step trajectories, identical seed and init:

  | step | all/uniform (default) | resolved/uniform (predicted harder) |
  |---|---|---|
  | 0 | .305 | .322 |
  | 60 | .492 | .493 |
  | 80 | .700 | .831 |
  | 120 | .632 | .678 |

  It climbs FASTER, ahead from step 70 on. In hindsight obvious: the state
  already contains that fact, so the imagined next state barely moves — a
  very distinctive geometric signature. A premature intent at least produces
  a novel-looking sentence.
  HONEST READING: the curves are noisy (default swings .700 -> .556 -> .632
  over 40 steps) and the .678 vs .632 gap at step 120 sits inside that band.
  The defensible statement is "no evidence `resolved` is harder, and the
  trend runs the other way". Do NOT launch it as the harder-negative arm.
  Variant #2 (late depth bias) also looks weak: mean insertion depth moves
  only 2.08 -> 2.35 because most rollouts are short. Wave 2 (`all/late` to
  isolate the depth lever, `premature/uniform` as the control) is running.
  RECOMMENDATION IF DEPTH STILL FAILS AT EPOCH 2/4: skip both variants and
  change the rollout policy — make rollouts follow the REMAINING TRUE
  SOLUTION TRACE instead of sampling feasible actions uniformly at random.
  Today the "continuation that occurred" at depth >= 1 is a coin flip among
  feasible actions, which is precisely why every available negative can only
  be a no-op. That is the root limitation, not the negative sampler.
- **Four-variant difficulty ordering settles it** (120 steps, identical seed
  and init; lower accuracy = harder task, commit b8b3509):

  | cf_kind / depth_bias | mean acc @ steps 80-120 |
  |---|---|
  | **premature / uniform** | **.507** (HARDEST) |
  | all / uniform (default) | .609 |
  | all / late | .613 (no effect) |
  | resolved / uniform | .710 (EASIEST) |

  The switch built as a throwaway CONTROL, `cf_kind=premature`, is the hard
  negative: ~10 points below default, and its curve FLATTENS after step 80
  (.553 -> .464 -> .437 -> .467) while the others keep climbing. That is
  exactly the "starts at chance, climbs slowly" signature we were looking
  for. Mechanism is defensible: a premature intent references variables
  whose parents are not yet resolved, so separating it requires tracking
  what is currently known — plausibly the judgement planning needs.
  The depth-bias lever does nothing measurable (.613 vs .609); it only moved
  mean insertion depth 2.05 -> 2.28 because most rollouts are short (horizon
  drawn from [1,2,4,8] and a rollout stops once solved). Not worth an arm.
  CAVEAT: 120 steps, one seed, within-run swing ~+/-.10 at fixed step. The
  premature-vs-resolved gap (.507 vs .710) is comfortably outside that; the
  all-vs-late difference is NOT resolvable.
  PLAN IF DEPTH STILL FAILS AT EPOCH 2/4: launch one arm with
  `cf_kind=premature, depth_bias=uniform`. Drop `resolved`, drop the depth
  bias. If that is still not enough, go to the rollout-policy change.

## 2026-08-20 — real looped-LM baseline: looping saturates, and loses

`looped-med-lr3e4-s0` COMPLETED (30 epochs, val_loss .3008). 200 episodes,
iGSM-med ID, same free-generation protocol, REAL `eval_loops` (not the dead
`looped_reread` substitute):

| K loops | success | invalid | unparseable | tok-pos/ep |
|---|---|---|---|---|
| 1 | .000 | 1.000 | .920 | 1255 |
| 2 | .000 | .692 | .197 | 4285 |
| 4 | .115 | .341 | .025 | 26205 |
| 8 | **.180** | .276 | .024 | 69360 |
| 16 | .170 | .279 | .020 | 140233 |

1. **Looping buys real accuracy and then saturates**: 1 -> 8 loops goes
   .000 -> .180, and K=16 gives nothing back for 2x the compute.
2. **It loses badly to the fixed-depth LM**: same lr, the ordinary 12-layer
   model reaches .81 free-gen at 29k tok-pos greedy. The looped model at its
   best is both worse AND ~2.4x more expensive.
3. FAIR CAVEAT to state in the paper: the looped model has 12.83M params vs
   90.80M — ~7x fewer. That was the deliberate design choice so extra loops
   could not be confused with extra capacity. The honest reading is
   "looping is not a substitute for depth here", not "looping is worthless".
PENDING before this goes in the paper: the better-converged twin
`looped-med-lr1e3-s0` (val .2287 vs .3008) is on its final epoch; its curve
should be the one reported, with lr3e-4 as the second seed.

- **The depth COLLAPSE is gone; depth PARITY is where we are.** Epoch-0
  backfill complete, epoch-2 depth-4 still running (running means, NOT
  final):

  | cell | epoch | interface | d1 | d4 |
  |---|---|---|---|---|
  | prefix4 | 0 | full_catalogue | .770 | .650 |
  | prefix4 | 0 | prior_propose | .620 | .400 |
  | prefix4-roll124 | 0 | prior_propose | .610 | .460 |
  | prefix4 | 2 | full_catalogue | **.980** | .967 (30/100 eps) |
  | prefix4-roll124 | 2 | full_catalogue | **1.000** | .860 (50/100 eps) |

  At epoch 0 depth clearly hurt (.770 -> .650, a .12 gap). At epoch 2 depth 4
  runs in the .86-.97 band against a d1 reference of .98-1.00. So the
  collapse the whole round was built to fix is gone — but depth 4 is at
  PARITY, not ABOVE, and "depth helps" is the claim we actually need.
  Sampling noise is still large (30 and 50 episodes; each episode moves a
  running mean 2-3 points). No conclusion until both finish.

- **METHODOLOGICAL PROBLEM, decide before epoch 4: depth 1 is at the
  ceiling.** .980/1.000 on `full_catalogue` leaves no headroom for depth to
  demonstrate a gain — a deeper search cannot beat a shallow one that
  already solves everything. Options:
  (a) rest the depth claim on `prior_propose`, which is far from ceiling
      (d1 .62 at epoch 0);
  (b) evaluate on a HARDER distribution — the OOD band, or faithful hard21 —
      where depth 1 is not saturated;
  (c) both, with full_catalogue reported as a saturated reference column
      (same status feasible_menu already has).
  Recommendation: (c). Note this is the SAME trap as feasible_menu being
  degenerate — a near-ceiling column cannot carry a comparative claim.
- **CORRECTION to the looped-LM entry above.** That table came from the
  worse-converged twin. The better one, `looped-med-lr1e3-s0` (COMPLETED, 30
  epochs, val_loss .2287 vs .3008), is much stronger and is the row to
  report:

  | K loops | success | invalid | unparseable | tok-pos/ep |
  |---|---|---|---|---|
  | 1 | .000 | .877 | .399 | 1215 |
  | 2 | .020 | .575 | .161 | 6004 |
  | 4 | .545 | .097 | .011 | 83854 |
  | 8 | **.730** | .050 | .004 | 216718 |
  | 16 | .720 | .053 | .006 | 414389 |

  So looping IS an effective test-time-compute strategy (.000 -> .730), not
  the weak one .180 suggested. My "loses badly" framing was premature —
  drawn from the worse seed before the better one finished.
  What stands: looping SATURATES HARD (K=8 -> 16 buys nothing for 2x the
  compute — a clean result in itself), and it still does not beat plain
  depth: .730 at 217k tok-pos vs the fixed 12-layer LM's .81 at 29k greedy,
  i.e. ~7.5x the compute for less accuracy, with ~7x fewer params.
  Against the JEPA planner at depth 1 (.94 at 9.6k tok-pos): better accuracy
  at ~1/23 the compute.
  Report lr1e-3 as the headline looped row and lr3e-4 as the second seed.

- **OOD does NOT solve the ceiling problem on `full_catalogue`.** Epoch-2
  checkpoints, 100 episodes, OOD = 16-21 ops vs 3-15 in training:

  | cell | ID d1 | OOD d1 | OOD invalid | OOD random |
  |---|---|---|---|---|
  | prefix4 | .980 | **.960** | .269 | .520 |
  | prefix4-roll124 | 1.000 | **.940** | .279 | .520 |

  The distribution shift costs depth 1 only 2-6 points. At .94-.96 there is
  still no headroom for depth to show a gain, so OOD is NOT a strong enough
  lever on this interface. Note random scores .520 here: `full_catalogue` is
  intrinsically forgiving, which is why it saturates.
  DECISION: `full_catalogue` joins `feasible_menu` as a SATURATED REFERENCE
  COLUMN, in-distribution and OOD alike. It cannot carry the depth claim.
  The depth claim rests on `prior_propose` (menu-free), where depth 1 was
  .620 at epoch 0 — far from ceiling. OOD `prior_propose` is running.

- `ecf16-prefix4-premature-s0` launched on gruenau9:2 (A100 80GB, verified
  empty) from snapshot 9d5dc3b with `cf_kind=premature, depth_bias=uniform`
  — the one lever the four-variant comparison supported. 0.08 s/problem,
  `energy_prefix_acc` .49 at step 40. Its watcher evaluates depths 1 and 4 on
  both interfaces, ID **and OOD**, every 2 epochs (depth 8 excluded, ~18 h).

- **Final epoch-2 depth pair, ID `full_catalogue`, roll124 arm, 100 eps:**

  | depth | success | invalid rate | mean steps |
  |---|---|---|---|
  | 1 | 1.000 | .037 | 7.2 |
  | 4 | .930 | .286 | 11.5 |

  Depth 4 below depth 1 — but depth 1 is at 1.000, so this column CANNOT
  show a gain. Uninformative for the claim, not evidence against it.
  THE DIAGNOSTIC IS THE REAL CONTENT: at depth 4 the EXECUTED invalid-action
  rate is .286 vs .037 at depth 1, and the planner takes 11.5 steps vs 7.2.
  So training `energy_prefix_acc` reached ~.99 while deeper search still
  executes wasteful steps at eval time. Training accuracy is saturating
  WITHOUT transferring to search — exactly the failure mode flagged when the
  accuracy hit .99, and the reason the `premature` (harder-negative) arm is
  the right next move rather than more weight on the current term.

- **FOUND THE UNSATURATED COLUMN: OOD `prior_propose`.**

  | interface / distribution | depth-1 success |
  |---|---|
  | full_catalogue, ID | .980 / 1.000 |
  | full_catalogue, OOD | .960 / .940 |
  | prior_propose, ID | .880 |
  | **prior_propose, OOD** | **.350 / .425** (40/100 eps, partial) |

  Depth 1 at ~.35-.43 leaves ample headroom, and `prior_propose` is the
  interface-matched, menu-free one reviewers take most seriously.
  PLAN: re-centre the depth claim on OOD `prior_propose`; `full_catalogue`
  (ID and OOD) becomes a reference column alongside `feasible_menu`.
  CHECK BEFORE COMMITTING: confirm the .35-.43 is bad SCORING and not the
  planner running out of proposals — the 08-20 diagnosis traced much of the
  earlier prior_propose collapse to exhausted proposal masks. Need
  `proposal_recall` and `no_proposal_episode_rate` at this difficulty.

- **Config-inheritance bug found and fixed (commit before snapshot
  `ee0b9b2e5bcace86fddf36326fb45a5211f78edd`).** `configs/flat_jepa_hard21.yaml`
  and `configs/flat_jepa_stylized.yaml` are STANDALONE — they do not inherit
  from `flat_jepa.yaml` — so they never received `latent_rollout_pred`, and
  any run using them on current code aborts instantly with
  `KeyError: missing config key objective.latent_rollout_pred`.
  This would have bitten at the worst possible time: those are the configs
  for the FINAL faithful-hard and stylized runs. Both now carry the key at
  weight 0; all three configs verified to parse with both new terms off, so
  nothing existing changes.
  WATCH FOR THIS: any new objective term must be added to all three configs,
  not just `flat_jepa.yaml`.

- Fifth fd-bug casualty: `flat-hard21-scratch-s0` (crashed ep1 step 10320,
  `RuntimeError: Too many open files`). Its `final_id` evals on best.pt
  survive. Relaunched on gruenau12:2 from snapshot ee0b9b2, training cleanly
  with `latent_rollout_pred=0.0000` as expected. Old dir kept as
  `-crashed-fd`.

## 2026-08-20 (night) — owner questions forced four corrections

**1. `full_catalogue` is not merely saturated, it is TRIVIAL.** Measured on
iGSM-med with the real config (`distractor_prob: 0.0`), 60 val problems:

| quantity | value |
|---|---|
| catalogue size | 12.3 (min 3, max 24) |
| of which EVER legal (in the sketch) | 12.3 — **all of them** |
| necessary steps | 5.9 |
| legal at the first step | 3.4 |
| attempt budget (4x necessary) | 23.5 |

Plus, from the code: illegal picks are masked permanently, so guessing is
sampling WITHOUT replacement (enumeration, not gambling);
`FaithfulEnv.step` calls `p2.to_sol`, i.e. **the environment computes every
outcome** — a policy never generates arithmetic; and `solved` just means the
query variable got resolved, with no answer string produced.
So random at .52-.59 is not "a correct chain of thought by luck": it is
"shuffle ~12 pieces, discard the ones that do not fit yet, retry, and let the
environment do the maths". THE INTERFACE MEASURES ORDERING, NOT SELECTION.
This retroactively explains feasible_menu .98 and the depth-1 ceiling.
ACTION: round `<date>-interface-difficulty` launched — turn distractors on,
enlarge the catalogue, find the level at which random stops being
competitive, and re-measure OUR model there. Expect our number to fall too;
the decisive quantity is the GAP to random.

**2. Averaged-prefix energy is not theoretically free** (owner question).
At fixed rollout depth mean = sum/H with H constant, so there is no length
artefact between candidates and the shared root cancels. Summing a
distance-like energy along a path = area under the distance curve, which
favours fast descent AND a close endpoint — the intended direction.
BUT: potential-based shaping telescopes to (endpoint - start), so ANY
aggregation that differs from endpoint-only is by construction NOT
potential-based and CAN change which plan is optimal. Specifically it
penalises NECESSARY DETOURS. Safe in iGSM (progress is monotone); NOT safe
in Blocksworld/ALFWorld, where un-stacking is required. Flag before porting.
CLEANER ALTERNATIVE to test: an illegal action is a no-op, so a wasted step
is one where the state does not move — penalise near-zero state movement
directly. Targets waste without assuming monotone progress and needs no
symbolic label. Add as a third `--aggregate` option.

**3. The probes measure the text, not the training** (owner was right that
.912 from an untrained net makes no sense). From
`data/probe_battery_stylized.json`: the quoted numbers are **AUC of a
nonlinear probe**, not accuracy; majority-class floor is **.779**
(`resolved`) and **.613** (`feasible`); in accuracy the arms are .833
trained / .819 untrained / .779 floor. The existing `mlp_shuffled_state`
control — destroy the state, keep the candidate action — still scores
**.720 AUC**, so most signal is in the ACTION not the state. And
"random-init" is NOT random numbers: it is an untrained deterministic
encoder of real text, and probes recover a lot from a random projection.
ACTION: round `<date>-probes-v2` — report every probe against BOTH floors
(majority-class and action-only), add state-only targets the action cannot
leak, balance classes, keep all controls.

**4. "Rollouts wander randomly" meant TRAINING DATA, not planning** (owner
confusion my fault). `src/textjepa/data/faithful.py` picks
`nxt = feasible[rng.randrange(len(feasible))]` when generating rollouts.
Eval-time planning does use energy-guided beam search (as of today's fix).
NOTE: `geo_rank_policy: greedy` and `ga_latent_beam` rollout policies exist
in the OLD discourse model (`_greedy_geo_labels`, `_latent_beam_geo_labels`)
but were NEVER ported to `flat_intent_jepa.py` — grep count 0. So the
current architecture only ever sees random rollouts. Making training
rollouts follow the solution is re-enabling a dropped mechanism, not a new
one.

**Launched in response** (4 rounds): detached proposal head (stop-gradient
so `intent_prior_lm` keeps the proposer without suppressing the geometry,
watcher reports BOTH planning and geometry in the same run); interface
difficulty; probes v2; learned action prior via flow matching (agent
instructed to first report why the codebook proposer failed — recall 1.0,
planning .027 — and to say so if a flow prior cannot address the real
failure mode).

- **Provenance confirmed: we ARE on the official iGSM implementation.**
  `src/textjepa/data/faithful.py` is a thin adapter over
  `facebookresearch/iGSM` (MIT), vendored at `third_party/iGSM`. Problems,
  prompt text, question, solution steps and answers all come from their
  `IdGen`; our additions are interface-only (planning interface + batch
  schema). Re-verified today: `scripts/validate_faithful.py 30` reports
  **official-checker pass 30/30, answer consistency 30/30**. The paper's
  headline environment is their generator, not a reimplementation.

- **Owner challenge to `mean_prefix`, and it is probably right.** Since every
  candidate sequence is rolled to the SAME depth, a wasted first step buys
  one FEWER real step of progress, so the plan should end further from the
  goal and ENDPOINT scoring ought to punish it unaided. The apparent
  endpoint-blindness was plausibly an artefact of the RANDOM TAILS — a
  wasteful root could draw lucky continuations while a clean root drew junk
  — which fix #1 (energy-guided beam expansion) already removed. The two
  changes were never separated.
  ROUND LAUNCHED to settle it: the 2x2 {random tails, energy beam} x
  {endpoint, mean_prefix} at depths 1 and 4, both interfaces, 200 episodes.
  If endpoint no longer loses once expansion is energy-guided, REVERT the
  default to `endpoint` and record that the averaging was unnecessary.
  Also adding a third option `movement`: penalise steps whose imagined state
  barely moves (an illegal action is a no-op, so waste is directly visible
  geometrically) — no goal-distance assumption, so unlike averaging it does
  not penalise necessary detours, which matters for Blocksworld/ALFWorld.
  FRAMING (owner): step-efficiency should be ONE result, not the lens on
  every experiment — nobody evaluates chain-of-thought on how fast it got
  there. Success rate is the primary metric; "solves in fewer steps" is a
  separate figure. Adopt this in the paper.
  COROLLARY once distractors are on: the dominant failure becomes picking a
  USELESS action, not wasting a step, so a length penalty should matter even
  less. Another reason to expect `endpoint` to be sufficient.

- **Gap in our diagnostics (owner):** we report the energy head's LEGALITY
  separation by imagined depth (d0 .818, d1 .940, d2 .910, d3 .864, d4 .838,
  d6 .791, d8 .734) but legality is not GOODNESS. Added: measure whether,
  among actions that are all legal, lower energy actually means the action
  ADVANCES the solution, at each imagined depth. Environment used only to
  LABEL the diagnostic (evaluation-only symbolic state, labelled as such);
  never a model input. This says whether the energy head can guide deep
  search at all, independently of how we aggregate.

- **Codebook proposer promoted to a first-class thread (owner):** simpler
  than a learned density and therefore more defensible. Historically recall
  1.0 with planning .027 — but the 08-20 diagnosis found menu-free episodes
  die by EXHAUSTING PROPOSALS and giving up, and separately that the old
  energy head was anti-feasible. Both would sink codebook planning while
  leaving recall at 1.0. Today's fixes address both, so re-running
  `codebook_ground` on a current checkpoint is the cheap first test and is
  to be done BEFORE building any density model.

## 2026-08-20 (night) — interface audit: I WAS WRONG; the BUDGET was the artefact

Full report: `research/reports/intent_phrase/2026-08-20-interface-difficulty/`
(mirrored to the paper repo). Checkpoint `flat-lminit-ecf16-s0/model/best.pt`
(epoch 2), copied before reading. 200 ID val episodes.

**Two of my premises from the entry above are FALSE. Corrections:**
1. "Every catalogue action is in the solution sketch — no wrong actions, only
   wrong orderings" is **WRONG**. Catalogue mean 11.7 vs solution length mean
   6.18 → **40.3% of catalogue actions are off-path**, and random's own
   executed `distractor_rate` is .35. The interface DOES test action
   selection. My error: I equated `p.sketch` membership with being on the
   solution path; it is not.
2. "Illegal picks are masked permanently, so guessing is enumeration" is
   **WRONG as stated**. The mask is cleared on every SUCCESSFUL step
   (`attempted = set(env.resolved)`), so it is sampling-without-replacement
   only within a run of consecutive invalid picks.
3. Also: `distractor_prob`/`max_distractors` CANNOT be the difficulty lever.
   They are consumed only in `FaithfulDataset.__getitem__` to push the
   TRAINING trace off-path; they never touch the generator, the catalogue or
   evaluation (`plan_flat.py` hardcodes 0.0 for eval). The distractor sweep I
   asked for would have produced six identical rows.
   My remaining premises (small catalogue, env computes outcomes, success =
   query resolved, 4x budget) all check out.

**The real artefact is the STEP BUDGET.** Difficulty settings barely move
random (.46-.59 at cap 4 across every generator setting tried; larger
problems are EASIER — useless fraction .403 -> .206 in the OOD band, which
also kills the "OOD is harder" claim). The budget moves random from .565 to
.000. Since no policy reads its budget, one cap-4 run scores every smaller
cap exactly from `solved_at`.

`full_catalogue`, ID, 200 episodes, depth 1:

| cap_mult | model | random | first-feasible | gap |
|---|---|---|---|---|
| 1.0 | .740 | .000 | .000 | **+.740** |
| 1.25 | .855 | .010 | .015 | **+.845** |
| 1.5 | .895 | .035 | .030 | +.860 |
| 4.0 (what we have been reporting) | .995 | .595 | .430 | +.400 |

**THE FEARED NEGATIVE DID NOT HAPPEN — the opposite did.** Tightening the
budget costs the planner .995 -> .740 while costing random .595 -> .000, so
the gap roughly DOUBLES. A harder distribution (`nec_range [3,5]`, useless
fraction .547) costs the model almost nothing (.850 at cap 1.25 vs .895).

**DECISION: keep ID + `full_catalogue`; change `--cap-mult` 4.0 -> 1.25, with
1.0 as a strict column.** Depth-1 headroom: .855 at 1.25, .740 at 1.0 — so
run the depth comparison at cap 1.0.
`scripts/plan_flat.py` gained `--nec-lo/--nec-hi`.

**Must relabel or drop:** every `feasible_menu` number (random .975 /
first-feasible .990 at cap 4; still .545 at cap 1.5); the .94-1.00 depth-1
`full_catalogue` headlines (real, but the margin over guessing was
UNDERSTATED ~2x and the ceiling made "depth helps" unmeasurable); "OOD is
harder" (it lowers the useless fraction); and any "random is at chance"
phrasing.

- **Detached proposal head implemented and launched** (round
  `runs/autonomy/intent_phrase/2026-08-20-detach-lm-state-v1/`, snapshot
  `_code/b8cdf47-detach-lm-state`). Switch
  `objective.intent_prior_lm.detach_state`, default false, added to ALL
  THREE configs.

  **The subtlety that makes this non-trivial: detaching the state is NOT
  enough.** The LM head is `nn.Linear(d, V, bias=False)` with
  `head.weight = tok.weight` — the output matrix IS the encoder's input
  embedding table. So the CE gradient still lands on `encoder.tok.weight`,
  which the trunk reads on every forward. And you cannot detach that too:
  the tied weight is the head's ONLY parameter, so detaching would make
  `intent_prior_lm` gradient-free everywhere and the proposer would never
  train. The only way to have both is to UNTIE. With `detach_state=true` the
  head gets its own matrix initialised as an exact copy of `tok.weight`, so
  training starts from the identical point.
  Measured at step 0, and note which half was bigger:

  | | detach arm | tied control |
  |---|---|---|
  | intent_prior_lm | lm_head 1.024 — **no encoder, no tok_embedding** | encoder .106 · **tok_embedding 1.023** |

  The trunk gradient (.106) was always the SMALLER half of this term's pull
  on the encoder; the larger half (1.023) went through the shared embedding.
  Detaching the state alone would have left that in place and the experiment
  would have looked done while changing almost nothing.
  Every other term's encoder gradient unchanged to 3-4 s.f.; flag-off is
  bit-identical (0/75 parameter gradients mismatched vs a git archive of
  HEAD).
  ALSO WORTH RECORDING: a first bit-identity check showed 27/75 "mismatches"
  that were a FALSE ALARM — multi-threaded CPU reduction order is
  nondeterministic at ~1.6e-7 relative scale; a same-code rerun mismatched
  on the same tensors. Pin to one thread for exactness checks.
  TRAP CAUGHT: `load_state_dict(strict=True)` would have SILENTLY loaded an
  untied checkpoint into a tied model — in a tied model `head.weight` and
  `tok.weight` are the same tensor, so the load succeeds while overwriting
  the embedding. `plan_flat.py` and `scripts/analysis/adapters.py` now read
  `detach_state` from the saved cfg and rebuild accordingly.

  Arms: `ecf16-prefix4-detachlm-s0` (gruenau7:3) and matched control
  `ecf16-prefix4-tied-s0` (gruenau12:3). Watcher reports BOTH sides in one
  pass every 2 epochs: depths 1/4 x prior_propose/full_catalogue x ID/OOD
  with `proposal_recall` and `no_proposal_episode_rate` (so a weak OOD
  prior_propose number can be distinguished from an exhausted proposal
  mask), PLUS the consequence-geometry AUC. First rows ~30 min out.

- **`scripts/rescore_budget.py` added — every plan JSON on disk can now be
  reported at ANY tighter budget with no reruns.** Valid because no policy
  reads its own budget: the episode at cap c is a strict prefix of the
  executed episode, so success at cap c is exactly
  `solved_at <= ceil(c * necessary)`. Verified to reproduce the interface
  audit's table exactly. Applied to all 350 plan files written since 08-19;
  only 5 carry reference-policy episodes because `episodes_random` /
  `episodes_first_feasible` were added just today (043b34f) — so this is
  retroactive only from that commit onward. **Every future eval gets the
  budget curve for free.**

- **THE BUDGET WAS INVERTING OUR MENU-FREE CONCLUSION.** From one early
  checkpoint (`ecf16-prefix4-premature-s0` epoch 0, step 500, 100 eps each) —
  absolute values are low because the checkpoint is young, but the pattern is
  unambiguous:

  | interface | cap 4.0 (reported) | cap 1.25 | cap 1.0 |
  |---|---|---|---|
  | prior_propose ID | model .650 / random .590 → **+.06** | .620 / .010 → **+.61** | .510 / .000 → **+.51** |
  | prior_propose OOD | model .140 / random .520 → **−.38** | .140 / .000 → **+.14** | .110 / .000 → **+.11** |
  | full_catalogue ID | .810 / .590 → +.22 | .210 / .010 → +.20 | .050 / .000 → +.05 |

  Read the middle row: at the budget we have been reporting, our planner looks
  **38 points WORSE than random** on menu-free OOD. At a realistic budget it is
  **14 points better**. The model's own score does not move at all
  (.140 → .140 → .110) — only random collapses.
  MECHANISM: our planner's successes are FAST; the reference policies only
  ever succeed by flailing inside a huge budget. A 4x budget therefore hands
  random almost all of its apparent competence, and every menu-free
  comparison we have reported is understated or outright inverted.
  This retires the earlier claims "prior_propose is barely above random"
  (.64 vs .59) and "at weight 16 it is .64 vs random .59, above random, not
  at it" — both were budget artefacts.
  CAVEAT: one early checkpoint, 100 episodes per cell. Re-confirm on mature
  checkpoints; but the direction is a property of the metric, not the model.
  ACTION: report the budget curve (1.0 / 1.25 / 4.0) for every planning table
  in the paper, with 1.25 as the headline and 1.0 as the strict column.

## 2026-08-20 (night) — proposers: the codebook revives, the flow does not help

Round `runs/autonomy/intent_phrase/2026-08-20-flow-prior-v1/`. New files:
`src/textjepa/planning/flow_prior.py`, `scripts/train_flow_prior.py`,
`scripts/measure_proposal_quality.py`, `tests/test_flow_prior.py`.

**Owner's codebook hypothesis CONFIRMED.** Re-running `codebook_ground`
unchanged on a current checkpoint (`flat-lminit-ecf16-roll124-s0`, with
`energy_cf_feasibility_rank` on), 100 val episodes, depth 1:

| | success | invalid rate | no-proposal episodes |
|---|---|---|---|
| codebook_ground | **.890** | .282 | **.000** |
| random (same rules) | .590 | .619 | — |
| first-feasible | .420 | .713 | — |
| historical | .027-.28 | — | ~100% |

Proposal exhaustion is gone and the planner beats its same-interface random
control by 30 points. **The codebook was never the bottleneck — the
anti-feasible energy head and the give-up behaviour were**, and both were
fixed today. .027 -> .89.

**BUT label it honestly: `codebook_ground` is CATALOGUE-PRIVILEGED.** It
snaps each k-means row to the nearest vector among the *current problem's own
catalogue* (`_filter_roots`), so its text is executable by construction and
its recall is near-1 by construction. With `prior_top_k=0` (the default) no
truncation happens at all, so on iGSM-med it kept ~12 of ~12 catalogue
actions — in the planning runs it was close to a re-ordering of the full
catalogue. It is menu-free only in the sense of having no feasibility oracle.
**`codebook_free` (genuinely catalogue-free) scores exactly .000 on every
axis at every K**, reproducing the 2026-08-11 result on a current
checkpoint. All of the codebook's planning strength comes from the catalogue.

**Proposal bench at matched K** (611 states, 100 val problems; oracle used
for measurement only):

| arm | K | recall necessary | parse | unique/state | decodes/state |
|---|---|---|---|---|---|
| token_head (current) | 4 | **.959** | .300 | 1.20 | 4 |
| token_head | 16 | .989 | .102 | 1.63 | 16 |
| codebook_ground* | 4 | .216 | 1.00 | 3.92 | 0 |
| codebook_ground* | 16 | .368 | 1.00 | 8.20 | 0 |
| codebook_free | any | .000 | .000 | 0.00 | K |
| flow_rerank | 4 | .799 | .376 | 1.22 | **64** |

*catalogue-privileged.

**The proposer is NOT the bottleneck, again.** The existing token head
already contains a progress-making action in its top 4 for **96%** of states.
There is no headroom for a better proposer to buy planning accuracy. The
flow loses at matched K (.799 vs .959) for 16x the decodes, improves parse
rate (.376 vs .300) and does NOT improve diversity (1.22 vs 1.20).
KEY MECHANISM: the ~1.6 unique-proposals-per-state figure is a collapse in
the TOKEN HEAD'S SAMPLING DISTRIBUTION, not in the selection rule — 64
oversampled phrases collapse to a handful of distinct strings before any
selection sees them. So no reranking machinery on top can fix it. The lever
that would: raise decode temperature and let a density gate filter the junk.
One-line knob, deliberately not added mid-round.

**Why a catalogue-free proposer is structurally hard here** (2026-08-12
diagnosis, re-confirmed): compute phrases name three variables, so 0/1367
held-out compute phrases were ever seen in 4000 training problems and the
phrase space grows linearly forever. An unseen action's EMBEDDING is nearly
reproducible (NN distance .77 vs 10.69 between same-problem actions) but the
embedding is a coarse role/operation code that does not carry the names, so
the decoder emits a different problem's phrase. This is a property of the
environment, not an engineering miss.

**RECOMMENDATION (agent's, and I agree): carry neither the flow nor the
catalogue-free codebook as the menu-free story.** The defensible menu-free
claim is catalogue enumeration from the prompt + energy-based selection. The
newly-earned result is that today's fixes lift codebook planning .027 -> .89.

**Two measurement bugs the agent caught in its own work**, both worth the
pattern: (1) the bench's `codebook_ground` arm first reported recall 1.0 /
parse 1.0 / unique 12.19 at EVERY K — `_filter_roots` dispatches on
`planner.candidate_interface`, which the bench left at default, so it was
silently measuring the whole catalogue. Uncorrected, this would have been a
FABRICATED CONFIRMATION of the recall-1.0 story. (2) a unit test caught a
diversity rule dominated by the density term that would have returned four
copies of one cluster.
Opt-in verified: zero new config keys (the flow is a separate artifact loaded
by `--flow-prior`, trained post-hoc on a frozen checkpoint, val NLL/dim
.743 -> -.184); all three interfaces produce BYTE-IDENTICAL result JSON
before/after; 8 new tests pass; suite 999 passed / 2 failed, both
pre-existing and unrelated (one fails standalone on main, one is a
test-order global-state flake).

## 2026-08-20 (night) — aggregation 2x2: owner hypothesis REFUTED; the energy head cannot rank goodness at depth

Round `runs/autonomy/intent_phrase/2026-08-20-aggregate-ablation-v1/`.
`ecf16-prefix4-roll124-s0` `last.pt` (copied first), 200 val episodes/cell.

**Success at depth 4** (depth-1 reference: full_catalogue 1.000,
prior_propose .880):

| interface | expansion | endpoint | mean_prefix | movement |
|---|---|---|---|---|
| full_catalogue | random (legacy) | .665 | .775 | — |
| full_catalogue | **beam** | .735 | **.915** | .680 |
| prior_propose | random (legacy) | .385 | .455 | — |
| prior_propose | **beam** | .370 | **.455** | .395 |

1. **Endpoint STILL loses to mean_prefix once expansion is energy-guided,
   and the gap is LARGER with the beam** (.735 -> .915) than with random
   tails (.665 -> .775). The endpoint-blindness was NOT a random-tail
   artefact. The owner's argument — a wasted step must cost one real step of
   progress, so the endpoint should already punish it — is sound in
   principle, but the imagined endpoint evidently does not encode that
   difference cleanly enough for the energy head to see it.
2. **The averaging, not the beam, is the load-bearing change.** With endpoint
   scoring, beam ~= random (.735 vs .665; .370 vs .385). The beam only pays
   off combined with mean_prefix.
3. DECISION: keep `mean_prefix` as the default; do NOT revert.
   PAPER CAVEAT to keep: mean_prefix is not potential-based and can change
   which plan is optimal, so the Blocksworld/ALFWorld port must RE-RUN this
   2x2 there rather than inherit the default.
4. `--aggregate movement` (endpoint + scale-free geometric penalty on
   rollout steps whose imagined state barely moves; no symbolic label, no
   goal-distance assumption) is implemented and tested but NEGATIVE: .680
   full_catalogue (worse than plain endpoint), .395 prior_propose. Kept as
   the only variant that survives the detour objection, but not a drop-in.

**THE MORE IMPORTANT RESULT — legality is not goodness, and goodness dies
past the root.** New `scripts/probe_flat_energy_progress.py` (ORACLE-LABELLED
DIAGNOSTIC: symbolic state labels the measurement only, never a model input).
Among LEGAL actions only, does lower energy mean the action is necessary for
the query?

| imagined depth | legality AUC | **progress AUC (legal only)** |
|---|---|---|
| 0 | .950 | **.842** |
| 1 | .990 | **.695** |
| 2 | .986 | .615 |
| 4 | .953 | .606 |
| 8 | .847 | .716 |

The head is a strong legality detector at EVERY depth but a weak goodness
ranker, and the goodness signal COLLAPSES the moment it leaves the root
(.842 -> .695 -> .615). **No aggregation can repair a score that cannot tell
a useful legal action from a useless one at imagined depth. This — not the
aggregation choice, not drift — is what blocks "depth helps".**
NEXT TARGET is therefore the energy head's training signal at depth >= 1,
which is exactly where the `premature` hard-negative arm and the
rollout-policy change (follow the true solution instead of random feasible
actions) both act.

**prior_propose depth-4 failure is proposal EXHAUSTION, not scoring:**
`no_proposal_episode_rate` .55-.63 at depth 4 vs .12 at depth 1. Deep search
executes ~10x more invalid actions (.15-.18 vs .017), they get masked, and
the planner runs out of candidates.

**Operational win: `--no-beam-diagnostics`.** The oracle "does the beam offer
options closer to the solved state than depth 1" MEASUREMENT (a full
re-encode of a completed trajectory per step plus three extra scoring passes)
was ~99% of depth>1 eval cost. Skipping it gives bit-identical plans and took
depth-4 evals from hours to minutes — which also makes the previously-18h
depth-8 eval cheap, so the depth-8 column is back on the table.

## 2026-08-20 (night) — probes v2: withdraw the v1 table

Report: `research/reports/intent_phrase/2026-08-20-probes-v2/`
(REPORT/METHOD/TABLES + data), mirrored to the paper repo. New code:
`scripts/analysis/probe_battery_v2.py`, `probe_domains.py`,
`summarize_probes_v2.py`, `common.py` additions, 7 new tests (27 pass).

DESIGN: balanced accuracy (floor exactly .500); state-only multi-label
targets (`resolved_set`, `frontier_set`) plus `query_reachable` /
`frontier_size` where an action-only baseline is at chance BY CONSTRUCTION;
controls `majority`, `prompt_only`, `step_index_only`, `prompt_plus_step`,
`shuffled_state`, `action_only`, `prompt_plus_action`, `state_only`; a
same-architecture-untrained arm for EVERY arm including both LMs; gradient
steps held constant across row counts so a 768-d state probe is not
handicapped against a 25-d control; and RANDOMISED feasible trajectories so
"problem + step index" does not determine the answer — that is what makes
`step_index_only` a real floor rather than a formality.

**FINDINGS, blunt:**
- The v1 headline is gone. .969/.912 were MLP AUC. On balanced accuracy
  against an ACTION-ONLY floor the trained stylized JEPA is **+.207**, its
  own UNTRAINED control **+.136** — only about **+.07 is training**.
- On `feasible` the SENTENCE LM (+.176) beats the trained JEPA (+.172), and
  the UNTRAINED sentence LM (+.153) beats the trained token LM (+.150).
- **Set-valued state readout is a flat negative for every arm in both
  domains**: `resolved_set` best margin over the step-counter floor is
  **-.002** across 16 arms; `frontier_set` best **+.010**. Counting the steps
  beats every encoder, trained or not.
- Steps-to-go is dead on stylized on the evidence, not just by policy: every
  arm is BELOW the step-index floor (JEPA R2 .480 vs floor .560).
- `operator_from_displacement` is architectural: the best stylized arm is an
  UNTRAINED token LM (.998) vs trained JEPA (.991).
- Training DESTROYS arithmetic value decoding: R2 .35/-.01/-.08 trained vs
  .58/.49/.59 untrained.
- **The one real positive**: on faithful, on IDENTICAL weights, JEPA training
  moves `frontier_size` from **-.036** (the token LM, i.e. the LM-init arm)
  to **+.067..+.136**, and `query_reachable` from +.018 to +.036..+.084.
  Replicates across seeds s0/s1/s2; from-scratch and untrained arms are at or
  below floor, so here the LM init IS load-bearing (note this cuts against
  the geometry result, where scratch beat LM-init — different property).
  But `nocfrank` is the BEST arm on `query_reachable`, so the energy
  counterfactual-ranking term is NOT what produces it — the same negative as
  the geometry ablation. And the corresponding SET question is at floor: the
  model encodes how OPEN the frontier is, not WHAT is in it.
- Sanity check passed: the `lm_init_untrained` arm reproduces the token-LM
  row to three decimals, as it must.

**DECISION: withdraw the v1 probe table rather than rescue it.** On stylized
iGSM, JEPA states are not better than LM states — the 2026-08-12 conclusion
stands. The only defensible representation claim is the faithful before/after
on two scalar probes worth a few points, and it must be printed next to the
set-readout negative.

- Incidental fix: `scripts/analysis/adapters.py` had a pre-existing
  IndentationError (an uncommitted `lm_detach_state` insert at the wrong
  level) that made the whole analysis package unimportable. Fixed.

## 2026-08-20 (night) — owner questions: four corrections and the OOD mechanism

**1. COMPARABILITY PROBLEM, must be fixed before the paper.** `success`
means two DIFFERENT things in our two headline tables. In the planner
interfaces the model never writes anything — it SELECTS from an enumerated
list and the environment writes the sentence and does the arithmetic;
success = the query variable got defined; there is no answer string and no
end-of-sequence decision. In the LM baseline the model must WRITE every
phrase. That is why random scores .59 in one and ~.00 in the other, and the
owner is right that a generating random policy could never emit
"Answer: X" and stop. A budget change does not fix this.
FIX: report BOTH families on ANSWER CORRECTNESS with the model producing the
outcome — the code already supports it (`gen_outcome=model`,
`success_answer_rate`; `plan_lm.py:170` requires `answer_correct is True or
outcome_source == "env"`). Under that criterion random genuinely is 0.
Report the budget curve rather than arguing for one cap.

**2. Why OOD is EASIER — measured** (120 problems each, distractors off):

| band | catalogue | necessary | **useless frac** | budget@4x |
|---|---|---|---|---|
| ID (op 3-15) | 11.1 | 5.74 | **.407** | 23.0 |
| OOD (op 16-21) | 16.4 | 12.51 | **.181** | 50.0 |

Bigger problems have more quantities but the solution uses a much LARGER
SHARE of them: catalogue +48%, necessary +118%. So traps fall from 41% to
18% — per decision the big problem is EASIER — and the budget scales with
length too (23 -> 50). Longer chains do compound error over 12.5 steps vs
5.74, but the two effects roughly cancel: random .59 -> .52, model .98 -> .96.
**"Longer" is NOT "harder" here.** To make OOD genuinely harder, raise the
TRAP FRACTION: keep the solution long AND enlarge the graph around it
(problem size is controlled separately from solution length; the interface
audit already hit useless_frac .547 by constraining necessary relative to
catalogue). Test that as the new OOD band.

**3. CORRECTION to my own recommendation: `prior_propose` IS a
no-menu-at-all model and it is our strongest one.** It writes the phrase
token by token from prompt+history; `_propose` grounds by EXACT TEXT MATCH
against `env.fp.params` (all parameters, not feasible ones), which is only
a text->object lookup for execution — the same grading the LM baseline gets.
It never sees which actions are legal or even which exist. It scores **.620
ID at cap 1.25 vs random .010**, and its top-4 contains a progress-making
action for **96%** of states. So the ordering is: generative proposer >
codebook (weaker AND needs the list) > flow (weaker still at matched cost).
I was wrong to lump the generative head in with the two that failed; the
defensible menu-free claim is stronger than I said.

**4. The sampling collapse, precisely.** 16 samples at temperature 1.3
decode to only ~1.6 DISTINCT valid actions per state because the model's
next-phrase distribution is very peaked — sampling repeatedly returns the
same one or two phrases, and oversampling to 64 yields more copies, not more
variety. Reranking cannot help because the variety is destroyed BEFORE any
selection rule sees the candidates. The lever is a higher sampling
temperature plus a learned density used as a FILTER (not as a proposer) —
which is the one role the flow model is genuinely suited to.

**5. Beam width, for the record:** `max_expand=64` beams kept, `branch=4`
continuations each. Coverage is wide; the failure is in scoring, not search.

## 2026-08-20 (night) — owner: target architecture is latent planning + detached decoder

**Model size, measured** (`FlatIntentJEPA` at `configs/flat_jepa.yaml`):

| module | params | note |
|---|---|---|
| encoder | **90.80M** | 768 wide, 12 layers, 12 heads |
| teacher | 90.80M | EMA copy, not gradient-trained |
| predictor | 16.52M | |
| observed_action_decoder | 12.66M | |
| horizon_energy_head | 3.55M | |
| total | 214.33M | distinct trained ~123.5M |

**The encoder is exactly GPT2-small shaped and exactly the size of the LM
baseline (90.87M)** — it is literally initialised from it. So the headline
comparison is like-for-like; we are not testing a smaller model against
GPT2-small. Record this in the paper.

**`--scorer oracle_distance` IS NOT AN ORACLE** (owner challenged the depth
collapse). `_goal_vector` completes the problem to build the TRUE solved
state, then ENCODES it with our own model and ranks by L2 in the LEARNED
LATENT SPACE. Right destination, broken ruler — and probes-v2 showed the
representation cannot report WHICH variables are resolved (set readout at
floor), only roughly how many. So that row measures the REPRESENTATION, not
an upper bound on search. We have never measured the real upper bound.
LAUNCHED: true symbolic oracle (remaining-necessary-steps computed from a
cloned env; evaluation-only diagnostic, labelled) x true executed endpoints,
depths 1/2/4/8, plus beam widths 8/64/256 to rule out narrowness with
evidence. If it scores ~1.0 the search machinery is fine and 100% of the
loss is representation + energy — a clean publishable decomposition. If not,
the search procedure itself loses episodes and that is a bug hunt.

**Owner memory to verify**: earlier STYLIZED experiments allegedly had
oracle-distance planning at ~1.0 AND probes showing the state truly encodes
variables and relations. Search launched across the archive, all dated
reports, HANDOFF/RESULTS, run JSONs and the paper mirror. Instructed to
guard against the likeliest false-memory source: old rows labelled
`privileged_expert_replay_over_symbolic_current_feasible_menu` /
`shared_symbolic_current_feasible_menu` — a 1.0 under EXPERT REPLAY or a
SYMBOLIC MENU is a different protocol. Also to report the training amounts,
since undertraining is a live hypothesis (today's cells are epoch 1-3 of 10).

**OWNER DECISION — target architecture**: learned prior over action
representations (codebook preferred, flow as fallback) + rollout ENTIRELY IN
LATENT SPACE + energy selection + a DETACHED DECODER rendering planned
action vectors to text only at execution. Model never sees a menu, a list or
a feasibility signal. Detachment justified by today's measurement: the
generative term is the biggest suppressor of the state geometry (.797 vs
.938), so a decoder that backprops into the encoder pays that cost and a
detached one does not.
THE GATE, and the round is instructed to report it first and stop if it
fails: can a decoder CONDITIONED ON THE PROBLEM CONTEXT exactly reconstruct
held-out action phrases? Every previous catalogue-free proposer died at
DECODING (parse rates .00/.03/.13; `codebook_free` re-measured today at
exactly .000). Structural cause: compute phrases name THREE variables,
0/1367 held-out phrases seen in 4000 problems, and the action embedding
carries the ROLE but not the NAMES. The innovation to test is letting the
decoder COPY names from the prompt (pointer / cross-attention over prompt
tokens) so the action vector need only specify role and slots.
Also flagged to that round: the same weight-tying trap the detach work hit —
if the decoder shares any parameter with the encoder, detaching activations
alone leaves the larger gradient flowing through the shared weights.

**OWNER DECISION — success criterion**: the model must EMIT THE ANSWER
EXPLICITLY. This is the only definition under which the planner and the LM
are doing the same task, and under it a random policy is 0 at any budget.
Adopt `success_answer_rate` with `gen_outcome=model` as the headline metric
for BOTH families. Expected side effect: OOD becomes genuinely harder,
because all ~12.5 steps must be right and the answer produced, so
compounding is no longer cancelled by the lower trap fraction.

**OWNER DECISION — trap count should scale**: define distractors as a
FRACTION/MULTIPLE of the number of legal actions, so bigger problems get
proportionally MORE traps instead of fewer (measured today: useless fraction
falls .407 -> .181 from ID to OOD). Fold into the interface work.

## 2026-08-20 (night) — the "oracle scored 1.0 before" memory is a PROTOCOL CONFLATION

Exhaustive search of the archive, all dated reports, HANDOFF/RESULTS, 85
`latent_planner_oracle_goal/*/success` rows on disk, and the paper mirror.

**Oracle-goal-distance planning was NEVER ~1.0.** Every stylized number:

| protocol | D1 | D2 | D4 |
|---|---|---|---|
| latent oracle-goal, NON-privileged (`2026-08-04-intent-oracle-distance-depth-controls-v1`) | **.106** | .456 | .534 |
| latent oracle-goal + candidate-privileged symbolic FUTURE feasible menu | .168 | .542 | **.756** (best ever) |
| latent oracle-goal, geometry-only (`2026-08-03` rq3, 3 LRs) | .125-.190 strict | | |
| latent oracle-goal (`2026-08-05` mechanism, 4 seeds) | .160-.210 strict | | |

**The 1.0s in those same JSON files are DIFFERENT ROWS:**
- `/oracle/success = 1.0` is **ExpertReplayPolicy**
  (`observed_action_search.py:353`) — 1.0 by construction, printed next to
  `random_policy .068` / `first_feasible .254` as the bounds pair.
- `latent_planner_symbolic_distance` D1 = **1.0** is **exact symbolic graph
  distance on exact environment states** (`search.py:288-293` refuses to run
  without `simulator=symbolic`). Not latent at all — AND IT ALSO COLLAPSES:
  1.0 -> .534 (D2) -> **.144 (D4)**.

So **today's .860 at depth 1 is the BEST latent-oracle-distance depth-1
number this project has ever recorded** — .106-.21 stylized. There is no
regression. And the depth collapse is OLD and reproducible: it is present in
stylized under an EXACT SYMBOLIC metric on EXACT states (1.0 -> .144), which
independently supports today's conclusion that the problem is not drift and
not search coverage.
Historic stylized ~.91 numbers do exist (RESULTS.md finding 11,
`disc_rank_k2` .91 strict / .985 slack-2) but that is a THIRD protocol: the
learned ranking energy over the CURRENT FEASIBLE MENU.
`RESULTS.md:168` gives the honest contemporaneous figure: oracle-goal
planning **.655**; `RESULTS.md:47-60` has raw goal-distance planning at
.150-.425 @optimal, noting it "fails on the unregularized space".

**TWO FALSE SENTENCES WERE LIVE IN THE PAPER MIRROR — now corrected**
(`2026-08-06-competitor-energy-baselines/REPORT.md:44,90,155` and
`2026-08-07-negative-results-appendix/REPORT.md:396` claimed an "oracle-goal
ceiling 1.000" / "100% success"). A correction note is inserted at the top of
both, the false phrases are marked inline, and both are re-mirrored to
`/vol/home-vol2/ml/laitenbf/TextJEPA-paper/reports/` and committed in both
repos. These would have reached a reviewer.

**The probe memory is also wrong, and the defect reproduces on stylized.**
`2026-08-11-state-readout-controls` reported resolvedness MLP acc .939 /
AUC .982 with a majority floor .781 and a shuffled-state control — i.e. it
had majority-class and shuffled-state floors but **no action-only, no
step-index, and no untrained-encoder control**. And both its targets were
per-(state, candidate) BINARY questions, never the set-valued "which
variables are resolved" question the memory attributes to them. Probes-v2
re-ran the set readout ON STYLIZED: `resolved_set` best margin **-.035**
against a step-index floor of .753; `frontier_set` best **+.010**. So the
floor result is not a faithful-domain artefact — it reproduces on stylized,
on the old 256-d encoders, trained and untrained. Nothing to regress from.

**Undertraining is contradicted, rank it last.** The old stylized headline
used DiscourseJEPA at **7.27M params**, d_state 256, **30,000 problems**,
10 epochs (~300k presentations, ~9,375 updates) — a 12x SMALLER model on
LESS data — and still only reached .106-.21 at depth 1. Today: 90.80M
encoder, ~100k problems/epoch, .860 at depth 1. The gap runs the other way.

**One genuine stylized->faithful loss does exist** and is already recorded:
cycle-consistency feasibility is stylized-only (AUC .85 stylized vs .48 =
chance on faithful). That is real — it is just not the loss remembered.
- union-prior first launch FAILED instantly: ctx_heads default 6 does not divide ctx-d-model 512 (MultiheadAttention assert). Relaunched 2026-08-21 ~a.m. with --ctx-heads 8, same cell dir, gruenau12:4.
- union-prior COMPLETED (relaunch, gruenau12:4, ~68 min). Prior on off-path val: top1 .416 / top4 .688 (ctx-only .285/.395; off-path .340/.480) — union beats both, no on-path/off-path tradeoff.
  Proposal quality (no menu shown; feasible set for measurement only), K=64: recall_feasible .957, recall_necessary .838, recall_true_next .797, parse .598, 0 states with no proposal.
  Depth-1 planning, 200 episodes: env success **.745**, answer-emission success **.695**, exact-necessary .477, no-proposal episodes .110.
  Ladder: eval-k32 .215 -> ecf16-retrain/wideK .47 -> ctx-prior .59 -> offpath .72 -> **union .745** (answer .19 -> .42 -> .52 -> .63 -> **.695**).
  Bounds this cell: random .595 env/.280 answer, first-feasible .430/.195 — planner clears both decisively on the answer criterion.
  Next: union prior is the new default proposal interface; take it to depth 2.
- Depth-2 round launched with the union prior + off-path decoder (200 episodes, branch 4, prior-samples 64, cap-mult 4.0): `union-d2` (endpoints=imagined, the real oracle-free system, gruenau12:5) and `union-d2-trueend` (endpoints=true, DIAGNOSTIC — candidates executed in an env clone and the REAL state encoded, gruenau12:6). Compare both against union d1 .745 env/.695 answer: if imagined-d2 collapses but true-end-d2 does not, the depth failure is in imagined rollout endpoints, not search or the prior.
- Depth-2 round COMPLETED. Both cells fall below union d1 (.745 env / .695 answer), and the ordering is INVERTED vs the standing "imagined rollouts drift" hypothesis:
  | cell | env success | answer | exact-necessary | invalid-action | no-proposal eps |
  |---|---|---|---|---|---|
  | union d1 (imagined) | .745 | .695 | .477 | .395 | .110 |
  | union-d2 (imagined, real system) | .640 | .550 | .406 | .539 | .120 |
  | union-d2-trueend (DIAGNOSTIC, env-executed real states) | **.465** | **.285** | .043 | .610 | .170 |
  Proposal metrics are ~identical across all three (recall .642-.664, parse ~.50), so the code prior is NOT the cause of depth collapse.
  **Giving the search TRUE encoded endpoints makes it strictly worse** — .465 env is BELOW the random-policy bound (.595) and .285 answer barely clears random (.265). The privileged diagnostic losing to the oracle-free system rules out imagined-rollout drift as the depth-collapse mechanism.
  Mechanism confirmed in code: the counterfactual energy head is trained in `discourse_jepa.py:_geo_rank/_energy_cf_feasibility` on `out.preds` — PREDICTOR OUTPUTS only. It has never seen encoder-produced latents in its candidate slot, so true encoded states are off-distribution for it. Depth collapse is an ENERGY-HEAD calibration failure, not a rollout-fidelity failure.
  Caveat for writeup: `endpoints=true` is candidate-privileged (executes candidates in an env clone) and must be labeled as such; it is used here only as a diagnostic, never as a system.
  Next candidates: (a) train the energy head with encoder-produced states mixed into the candidate slot so imagined/true are interchangeable; (b) depth-2 ranking accuracy probe to measure energy mis-ranking directly rather than through end-task success.

### 2026-08-21 monotonicity measurement — the oracle_distance scorer was measuring in the wrong space

`scripts/measure_energy_monotonicity.py` (new), 200 val problems, ecf16-s0. Walks the ground-truth
solution, encodes every prefix, and ranks every feasible action at every true state. No planner,
no generation. ORACLE DIAGNOSTIC (goal + feasible sets from the env); measurement only.

**Root cause: predicted and encoded states are on different scales.** `latent_pred` uses
`norm_targets: true`, and `objectives/base.latent_distance` LayerNorms BOTH sides — so the predictor
is trained to match the encoder only up to LayerNorm, and raw scale/offset is unconstrained.
Measured: encoder state norm **66.6**, predicted state norm **147.8** (ratio **2.22x**).
But `flat_search._score` scores `oracle_distance` with RAW L2 (`(endpoint-goal).norm(dim=-1)`).

| statistic (200 problems) | raw L2 (what the scorer uses) | LN-L1 (the space the predictor is trained in) |
|---|---|---|
| true next action ranked top1 by distance | **.109** | **.822** |
| distance-argmin is a necessary action | **.185** | **.898** |
| predicted endpoint closer to goal than current state | **.000** | **.896** |
| descend_frac along true trajectory | .793 | .789 |
| Kendall tau of goal distance vs step index | -.799 | -.790 |

- **Monotonicity is NOT the problem.** The goal potential decreases at ~79% of steps with tau ~-.79
  in BOTH metrics. Hypothesis (i) "the potential is not monotone" is rejected.
- **The raw-L2 numbers land where the planner's oracle-distance arm lands**: argmin-is-necessary
  .185 here vs exact-necessary **.183** for `d1-full-lat-imag`. CAVEAT: these are DIFFERENT
  checkpoints (measurement on `_ckpt_snapshots/ecf16-s0-best-2026-08-20.pt`; the ladder on
  `2026-08-20-aggregate-ablation-v1/ckpt.pt`, md5s differ), so the numerical agreement is
  suggestive, not a reproduction. A within-checkpoint re-run is queued.
- **The learned energy head is fine at depth 1**: true-action top1 **.872**, argmin-is-necessary
  **.951**, percentile .077. It is an MLP over raw vectors, so it absorbs the scale offset; the
  distance scorer cannot.
- **This also explains `endpoints=true` being worst** (union-d2-trueend .465). The energy head is
  trained on PREDICTED endpoints (norm ~148); feeding it ENCODER states (norm ~67) is a 2.2x
  off-distribution shift in its candidate slot.

**Consequence for the record:** every `oracle_distance` row in
`2026-08-20-true-oracle-upper-bound` (the "latent oracle-goal ruler is blunt / collapses with depth"
conclusion) is contaminated by this metric bug and must be re-run in LN-L1 before being cited.
The `symbolic_oracle` and `energy` rows are unaffected.

**Actions taken (2026-08-21, same session):**
- `flat_search.goal_distance()` + `plan_flat --distance-metric raw|ln_l1` added (default `raw`
  preserves old behaviour). Round `2026-08-21-lnl1-ladder-v1` re-runs the oracle-distance ladder
  d1/d2/d4/d8 in LN-L1 on the ladder's own checkpoint (`2026-08-20-aggregate-ablation-v1/ckpt.pt`),
  same protocol as the raw rows (full_catalogue, max-expand 64, 200 episodes).
  NOTE: runs from the LIVE tree, the metric fix is not committed/snapshotted yet.
- Ported into the flat training path (all weight 0.0 by default, all ADD NO PARAMETERS so warm
  start stays strict): `objective.straighten` (TemporalStraightening),
  `objective.monotone` (GoalMonotonicity, label_free -- reads no necessary/distractor annotation),
  and a NEW `objective.energy_monotone` (`EnergyMonotonicity`, geometry.py): from one fixed root
  s_0, E(s_0, s_t, s_0, t) must fall as t grows. That is the cross-time constraint the survey
  confirmed does not exist anywhere -- every other Energy term compares only within one anchor,
  horizon and root. Also fixed `GoalMonotonicity(label_free=True)` touching `batch["necessary"]`,
  a key the flat pipeline never provides.
  CPU smoke (`runs/smoke_geo`) confirms all three produce gradients into the expected modules
  (energy_monotone reaches horizon_energy_head).
- Round `2026-08-21-geometry-sweep-v1` queued, 6 cells, each warm-started from the ecf16 run's
  `last.pt` and trained 6 more epochs, so every cell is also a train-longer arm:
  `longer` (control), `rawpred` (`latent_pred/counterfactual_state.norm_targets=false` -- the direct
  fix for the 2.2x scale gap, needs no new code), `straighten`, `monotone`, `energymono`,
  `rawpred-energymono`.
- Both rounds run behind a GPU-polling queue (`queue.sh`) because Gruenau was at 0/34 free GPUs;
  it dispatches one cell per genuinely idle GPU (VRAM <1.5GB AND util <10%).

**CORRECTED ORACLE LADDER (`2026-08-21-lnl1-ladder-v1`), same protocol and same
checkpoint as the raw rows** (full_catalogue, max-expand 64, 200 episodes,
`2026-08-20-aggregate-ablation-v1/ckpt.pt`). Runs from the LIVE tree; snapshot
before quoting as a headline.

| depth | raw L2 env / exact (2026-08-20) | LN-L1 env / exact (corrected) |
|---|---|---|
| 1 | .955 / .183 | **1.000 / .610** |
| 2 | .915 / .164 | **.985 / .371** |
| 4 | .920 / .038 | **.990 / .303** |
| 8 | .890 / .011 | **.980 / .352** |

Mean steps fall from 11.3-14.3 to 7.4-9.0. **The "latent oracle-goal ruler is
blunt and collapses with depth" result was a metric artifact.** Under raw L2
exact-necessary fell monotonically and catastrophically, 16x from d1 to d8
(.183 -> .011). Under LN-L1 it drops from d1 to d2 (.610 -> .371) and then
FLATTENS: .303 at d4, **.352 at d8** -- d8 is BETTER than d4, so there is no
depth collapse at all beyond depth 2, only a one-step drop. Env success is
.980-1.000 at every depth. What remains to explain is the single d1->d2 step,
not a collapse.

Docs: `docs/latent_metric_spaces.md` (durable rule).
Report: `research/reports/intent_phrase/2026-08-21-energy-geometry/REPORT.md`.

**Second sweep queued** (`2026-08-21-geometry-sweep-v1/cells2.txt`):
`quasimetric` (energy as a Metric-Residual-Network distance to a GoalHead
prediction -- triangle inequality by construction; adds params, needs
`init_allow_missing=true`), `hindsight` (goal relabeled to a random observed
future state, O(T^2) constraints instead of O(T)), plus
`quasimetric-energymono` and `hindsight-rawpred`. Both smoke-tested; hindsight
gradient flow verified 20/20 at realistic shapes after its grad-norm report
read zero on a 2-example CPU microbatch.


**Warm-start fix (2026-08-21)**: the `quasimetric` cell failed immediately —
`init_allow_missing` only tolerated MISSING keys, but replacing the energy head
also makes the OLD head's keys UNEXPECTED
(`init_from_ckpt mismatch: unexpected=[horizon_energy_head.net...]`).
`train_flat_jepa.py` now filters incoming checkpoint keys the current config
does not use (name or shape mismatch) and reports both lists. Verified
directly: an `mlp`-head checkpoint loads into a `quasimetric`-head model with 6
keys dropped, 18 missing (all the new head's goal/phi/psi), 0 unexpected.
Cell requeued via `queue3_chained.sh`, which waits for `queue2.sh` to exit so
two pollers never grab the same GPU.

**Verification (2026-08-21)**: every test file that touches the changed modules
(`flat_search`, `plan_flat`, `objectives.geometry`, `HorizonEnergyHead`,
`FlatIntentJEPA`, `train_flat_jepa`) passes — **107 tests, 0 failures**:
`test_energy_cf_feasibility` + `test_energy_prefix_rank` +
`test_flat_sentence_jepa` (18 passed, 8m28s), and
`test_symbolic_oracle_scorer` + `test_code_prior_interface` + `test_model` +
`test_flow_prior` (89 passed, 39m14s). `test_symbolic_oracle_scorer` is the
load-bearing one: it exercises the same `flat_search._score` path where the
distance computation changed. The `raw` default is therefore confirmed
behaviour-preserving.

### 2026-08-22 geometry sweep evaluated (`2026-08-22-geometry-eval-v1`)

Five cells, each warm-started from the crashed ecf16 `last.pt` and trained 6
more epochs (so all are also train-longer arms). Metric diagnostic = 200 val
problems; planning = full_catalogue + oracle_distance, 200 episodes
(CANDIDATE-PRIVILEGED MENU + ORACLE GOAL — diagnostic, never a headline).

| cell | enc norm | pred norm | ratio | nec@raw | nec@LN | ln_d1 env/exact | ln_d2 env/exact |
|---|---|---|---|---|---|---|---|
| `longer` (control) | 56.5 | 115.4 | 2.04 | .409 | .763 | 1.00 / .41 | .94 / .10 |
| `rawpred` | 26.9 | 25.6 | **0.95** | .553 | .603 | .92 / .20 | **.56 / .00** |
| `hindsight` | 110.2 | 132.3 | 1.20 | .164 | **.920** | **.99 / .66** | **.94 / .27** |
| `hindsight-rawpred` | 26.5 | 25.7 | **0.97** | .595 | .646 | .95 / .20 | .66 / .00 |
| `quasimetric-energymono` | 107.5 | 148.0 | 1.38 | .149 | **.922** | **.99 / .65** | .94 / .14 |

**1. `norm_targets=false` works as advertised and is still the wrong choice.**
It closes the scale gap exactly (ratio 2.04 -> 0.95) and, as predicted, that
lifts the RAW-metric ranking (nec .409 -> .553). But it *degrades* the
representation: LN-space ranking falls BELOW the control (.603 vs .763) and
depth-2 planning collapses (env .94 -> **.56**, exact .10 -> **.00**). VICReg
val was already elevated in exactly these two cells (.494/.503 vs .346/.348),
i.e. the anti-collapse term was visibly working harder. **Answer to "LayerNorm
at scoring, or train without it?": LayerNorm at scoring. Do not train without
it.**

**2. `hindsight` is the best cell, and it beats the pre-sweep checkpoint.**
nec@LN **.920** and d1 exact **.66**, against .898/.610 for the original ecf16
and .763/.41 for the train-longer control. Hindsight relabeling is the single
most effective addition tested.

**3. `quasimetric-energymono` matches it at d1** (.922 / .65) but is weaker at
d2 (.14 vs .27).

CORRECTION (2026-08-23): I earlier read its `energy_monotone` val hinge of
0.0000 as the quasimetric head satisfying monotonicity STRUCTURALLY. That
inference is wrong. In that cell the penalty was ACTIVE (weight 1.0), and the
plain-MLP `energymono` cell also reaches exactly 0.0000 — so the reading shows
only that the loss does its job, with either head. The standalone `quasimetric`
cell reads 0.0000 too, but there the weight is 0, the model flag is off, the
extras are absent and the objective returns a constant zero: a NO-OP, not
evidence. The structural claim is untested by these numbers. The statistic that
would actually test it is `energy_tau_vs_index` from
`measure_energy_monotonicity.py` on the standalone `quasimetric` cell (loss
never applied) versus `longer`; that eval is queued.

**4. Training longer alone HURTS.** The `longer` control went from the original
ecf16's nec@LN .898 / d1 exact .610 down to **.763 / .41** over 6 further
epochs. More steps is not the missing ingredient; it is mildly harmful on this
metric. That retires the train-longer hypothesis.

**5. The d1 -> d2 drop survives everything.** Best cell still falls .66 -> .27.
No geometry term tested closes it. This is now the single open question.

### 2026-08-23 geometry eval, 8 of 10 cells (planning rows CANDIDATE-PRIVILEGED)

| cell | pred/enc | nec@LN | top1@LN | E_tau | descend@LN | ln_d1 env/exact | ln_d2 env/exact |
|---|---|---|---|---|---|---|---|
| `longer` (control) | 2.04 | .763 | .698 | **-.579** | .704 | 1.00 / .41 | .94 / .10 |
| `straighten` | 1.23 | .920 | .829 | -.562 | **1.000** | .99 / **.68** | .94 / **.04** |
| `hindsight` | 1.20 | .920 | **.853** | -.404 | .748 | .99 / .66 | .94 / **.27** |
| `quasimetric` | 1.38 | .916 | .848 | **-.214** | .686 | .98 / .66 | .91 / .16 |
| `quasimetric-energymono` | 1.38 | **.922** | .851 | -.305 | .679 | .99 / .65 | .94 / .14 |
| `rawpred` | 0.95 | .603 | .525 | -.440 | .667 | .92 / .20 | .56 / .00 |
| `hindsight-rawpred` | 0.97 | .646 | .550 | -.418 | .808 | .95 / .20 | .66 / .00 |

**1. The quasimetric structural claim is REFUTED.** `energy_tau_vs_index` on
the standalone `quasimetric` cell — where `energy_monotone` was NEVER applied —
is **-.214**, markedly LESS monotone than the plain-MLP control's **-.579**.
The Metric-Residual parameterization does not buy monotone energy for free; it
is worse than the unstructured head. (This is the test that replaces the
0.0000-hinge misreading corrected above.)

**2. Perfect distance monotonicity does NOT help depth — it hurts it.**
`straighten` achieves `descend_frac` **1.000** (vs .704 control): the goal
potential decreases at EVERY step of every trajectory. It also gives the best
depth-1 exact-necessary (**.68**). Yet its depth-2 exact-necessary is the worst
of any non-collapsed cell (**.04**, below the .10 control). Straightening also
drove VICReg val to .583, the highest of any cell.

**3. Energy monotonicity is ANTI-correlated with planning quality here.** The
control has the most monotone energy (E_tau -.579) and the worst d1 planning
(.41); the best planners (`hindsight` -.404, `quasimetric` -.214) are the least
monotone. Together with (2), the monotonicity hypothesis is now doubly
refuted: neither distance-monotonicity nor energy-monotonicity predicts
planning quality, and the one intervention that maximized distance
monotonicity made depth-2 worse.

**4. `hindsight` remains the best overall cell** — uniquely holding depth 2
(.27 vs .04-.16 for everything else) while matching the field at depth 1.

**5. The d1 -> d2 drop is universal.** Every cell falls (best .66 -> .27). No
geometry term tested closes it.

### 2026-08-23 geometry sweep COMPLETE — all 10 cells (planning CANDIDATE-PRIVILEGED)

| cell | pred/enc | nec@LN | top1@LN | E_tau | descend | ln_d1 | ln_d2 |
|---|---|---|---|---|---|---|---|
| `longer` (control) | 2.04 | .763 | .698 | -.579 | .704 | 1.00/.41 | .94/.10 |
| `straighten` | 1.23 | .920 | .829 | -.562 | **1.000** | .99/**.68** | .94/**.04** |
| `monotone` | 2.26 | .895 | .825 | -.544 | .815 | .99/.61 | .94/.17 |
| `energymono` | 1.75 | .905 | .830 | **-.652** | .693 | 1.00/.64 | .91/.06 |
| `hindsight` | 1.20 | .920 | **.853** | -.404 | .748 | .99/.66 | .94/**.27** |
| `quasimetric` | 1.38 | .916 | .848 | -.214 | .686 | .98/.66 | .91/.16 |
| `quasimetric-energymono` | 1.38 | **.922** | .851 | -.305 | .679 | .99/.65 | .94/.14 |
| `rawpred` | 0.95 | .603 | .525 | -.440 | .667 | .92/.20 | .56/.00 |
| `hindsight-rawpred` | 0.97 | .646 | .550 | -.418 | .808 | .95/.20 | .66/.00 |
| `rawpred-energymono` | 0.95 | .607 | .526 | -.451 | .677 | .91/.18 | .62/.00 |

**Every geometry term beats the control at depth 1** (.61-.68 vs .41) and every
one lifts nec@LN (.895-.922 vs .763). The additions work; the control is the
weakest healthy cell. But at depth 2 only `hindsight` (.27) clearly beats the
control (.10) — `straighten` (.04) and `energymono` (.06) are WORSE.

**Monotonicity does not explain planning.** Across the 7 healthy cells
(excluding the 3 collapsed raw-target ones):
- `descend_frac` vs d1 exact **r=+.26**, vs d2 exact **r=-.30**
- `E_tau` vs d1 exact **r=+.38**, vs d2 exact **r=+.53** (LESS monotone energy
  goes with BETTER planning; the sign is the wrong way round for the
  hypothesis)
`straighten` reaches PERFECT distance monotonicity (descend 1.000) and has the
worst healthy d2 (.04). `energymono` has the MOST monotone energy (-.652) and
the second-worst d2 (.06). Neither n=7 correlation is significant, but nothing
here supports monotonicity as the mechanism, and the two cells that maximized
it are the two that lost depth 2.

**All three `rawpred` cells collapse identically** (d2 exact .00, env .56-.66)
while having the only closed norm ratios (0.95-0.97). Adding `energy_monotone`
does not rescue it. Confirms: fix the metric at scoring time, never retrain
without LayerNorm targets.

**Verdict: `hindsight` is the recipe change to keep.** Best top1@LN (.853),
tied-best nec@LN, and the only cell that holds depth 2. Everything else either
matches at d1 and loses d2, or degrades outright.

**Open question unchanged**: the universal d1 -> d2 drop (best .66 -> .27).

### 2026-08-24 monitor alert: `2026-08-20-prefix-energy-v1/ecf16-prefix16-s0` FAILED — mislabelled

**The cell did not fail. Its results are complete and usable.** Training ran the
FULL 10-epoch schedule (`[ep 9 step 62480]`, `training_complete.json`:
`status=completed, steps=62500, best_val_total=1.4047`), wrote `best.pt` and
`last.pt`, and every eval artifact was produced: 15 plan JSONs + `probe_auc` in
`final_id`, 6 in `final_ood`.

The `FAILED` label comes from `job.sh` recording `rc` of the TRAINING process
(`exit_code=1`), which is evaluated BEFORE the eval stage and is the only thing
that sets `state`. `stderr.log` is **2 lines long and contains only the
`enable_nested_tensor` warning** — no traceback, no OOM, no signal (a signal
would give 128+n, and `timeout` would give 124). So the nonzero exit happened
on the interpreter's exit path AFTER `training_complete.json` was written;
most likely dataloader-worker teardown with `num_workers=6`. Not a training
failure, and nothing to re-run.

Headline numbers from it (scorer=energy, so UNAFFECTED by the LN-L1 metric bug;
300 episodes; `full_catalogue`/`feasible_menu` are CANDIDATE-PRIVILEGED menus):

| interface | d1 | d2 | d4 | d8 | d16 |
|---|---|---|---|---|---|
| `feasible_menu` env/exact | 1.000/.830 | 1.000/.750 | .980/.269 | .973/.240 | .983/.237 |
| `full_catalogue` env/exact | 1.000/.817 | .970/.632 | .850/.169 | .740/.194 | .707/.222 |
| `prior_propose` env/exact | .923/.906 | .587/.915 | .467/.893 | .440/.871 | .440/.879 |

Note this is a full-schedule (62.5k step) run, i.e. the schedule ecf16 never
reached — and its d1 exact-necessary (.817-.830 on menus) is far above anything
in the geometry sweep. Recipe differs (`energy_prefix_rank.weight=16`), so it
is not a clean comparison, but it is the strongest d1 number on record and
deserves a proper head-to-head.

ACTION NEEDED (not done): `job.sh` in that round sets `state` from the training
rc alone, so a clean run with a noisy exit path is labelled FAILED while a run
whose EVAL stage dies is labelled COMPLETED. Both directions are wrong.

## 2026-08-24 — prefix rank largely solves d2; combo round launched

Full depth ladder of the completed `ecf16-prefix16-s0` (300 eps, energy scorer,
exact-necessary): full_catalogue **.817/.632/.169/.194** at d1/d2/d4/d8 (env
1.00/.97/.85/.74). The d1->d2 drop that no geometry term closed (best .66->.27)
is largely closed by `energy_prefix_rank.weight=16` at full schedule: **d2 .632**.
The open problem moves to d2->d4 (.632->.169, distractor rate .17->.36).
Separate failure on `prior_propose`: exactness stays ~.9 at ALL depths but env
success falls (.923->.587 at d2) via proposal recall (.976->.908) — a proposer
coverage problem, not an energy-ranking problem.

Committed + snapshotted the geometry/metric work (d5f2d81). Launched
`2026-08-24-prefix-combo-v1` on gruenau1 (3 GPUs + 1 chained), all warm-started
from prefix16 last.pt, 6 epochs, prefix_rank kept at 16, eval (d1/2/4/8 x
full_catalogue/prior_propose, 200 eps) baked into job.sh; state now derived
from training_complete.json + eval rc (fixes the false-FAILED labeling flaw):
- `p16-longer` — control (warm-start effect only)
- `p16-hindsight` — + hindsight_monotone 1.0 (geometry winner)
- `p16-late` — depth_bias=late (junk step biased deep, targets d4)
- `p16-resolved` — cf_kind=resolved (harder negatives)
ETA ~16h/cell; read results tomorrow morning.

## 2026-08-24 (later) — round moved to Alex rtxpro6k; 5th cell added

Grünau round superseded: p16-resolved OOM'd on gruenau1's 22GB RTX 6000
(needs ~24GB); relaunched on gruenau7 A6000s, then the whole round was
re-submitted to Alex (`sbatch --partition=rtxpro6k,a100`), where all cells
started within 30s on one RTX PRO 6000 Blackwell node at 3.06x V100
throughput (s_per_problem .176 vs .538) -> ETA ~5-6h. All 4 Grünau cells
killed (state=CANCELLED-superseded-by-alex). Alex staging in
`~/textjepa-0824` on the HPC home because the atuin group volume is over its
FILE-COUNT quota (573K/500K files) — needs cleanup. Alex does not share the
FS; a watcher on gruenau1 pulls each cell's last.pt every ~90 min, evals
full_catalogue d2/d4 (100 eps) on the freed V100s into `watch_curve.jsonl`,
and rsyncs everything back to `alex-final/` when the round completes.

NEW 5th cell `p16-rollpred` (`objective.latent_rollout_pred.weight=1.0`,
ks=[1,2,4]): the predictor currently has ZERO multi-step supervision (only
1-step latent_pred); prefix rank constrains multi-step imagined states only
through the energy. Latent drift over deep rollouts is the prime suspect for
the remaining d2->d4 drop (.632->.169), and this knob is the most direct,
already-implemented, self-supervised counter.

## 2026-08-24 (evening) — trajectory-diversity find; fresh-data cells launched

Found: the trajectory RNG is seeded per problem index (`{seed}:{index}:t`), so
every epoch replays the IDENTICAL solution ordering per problem — one ordering
per problem, ever. Likely explains "train longer hurts" (verbatim replay =
ordering overfit) and the deep-beam blind spot (valid unobserved orderings
produce unfamiliar encoded states, since states encode rendered step text).
Owner decision: no cross-ordering alignment loss for now (feels too symbolic);
fix via data diversity instead.

Launched on Alex (warm-start prefix16, energy_prefix_rank 16, same protocol as
prefix-combo round): `p16-freshdata` (data.train_size=1M, train.epochs=1 —
every trajectory unique, same compute shape as 10x100k) and
`p16-freshdata-rollpred` (+ latent_rollout_pred.weight=1.0). ~9h train + eval.

Diagnostics so far (all on prefix16 ckpt, eval-only): d4 drop is STATE DRIFT,
not energy (oracle LN-L1 ruler also fails at d4: .173 vs energy .169; d2 .367
vs energy .632 — energy BEATS oracle distance at d2); not search budget
(expand-256 at d4: .143); proposer collapse not fixable by sampling wider/hotter
(recall stuck ~.89-.90, env worse). MPC replans every step (D2 dead). Running:
A1 exposure-bias knockout (--endpoints true + oracle ln_l1, d1-d8), D1
aggregation sweep (endpoint/movement), oracle-d8, expand256-d8.

## 2026-08-24 (night) — ROUND VERDICT: exposure bias confirmed with a causal fingerprint

prefix-combo final (full_catalogue exact-necessary, 200 eps, d1/d2/d4/d8):
control .813/.636/.192/.197 (= baseline: extra compute changes nothing);
hindsight .825/.646/.200/.194 (no gain — obsoleted by prefix rank);
late .834/.635/.280/.206; resolved .809/.577/.265/.176;
ROLLPRED .803/.677/.305/.155 — near-doubles d4, best d2, and its gains stop
EXACTLY at its supervised horizon (ks=[1,2,4] -> gains at d2/d4, none at d8).
Causal fingerprint: depth drop = predictor exposure bias; supervising k-step
rollouts fixes exactly the depths supervised. late = smaller independent
energy-side gain, only cell nudging d8. Proposer env ceiling (~.57 at d2)
unchanged in every cell — separate prior-coverage problem.

Launched: p16-rollpred-k8 (ks=[1,2,4,8] — prediction: gain moves to d8) and
p16-rollpred-late (stack predictor+energy fixes). Queued behind the running
freshdata pair. Plan JSONs synced to prefix-combo-v1/alex-final/.

## 2026-08-24 (late) — B1 + C1 built and launched; full board saturated

B1: `energy_prefix_rank.n_insert` (multi-insert counterfactual paths — waste
2-3 steps instead of 1, covering deep-beam-divergent negatives). C1:
`plan_flat --root-agg mean` (first action chosen by MEAN energy of each
root's surviving beams instead of its luckiest rollout — variance reduction
against the winner's curse). Committed 2863250, tests green (4+7).

Board: rtxpro6k node saturated with 8 training cells (freshdata,
freshdata-rollpred, p16-rollpred-k8, fresh-rollpred-k8, fresh-rollpred-late,
fresh-solprob, fresh-horizon, fresh-causal-rollpred); a100 partition:
fresh-insert2 / fresh-insert3 / fresh-insert2-rollpred-k8 pending. Grünau:
A1 knockout (endpoints=true oracle d1-8), D1 aggregation sweep, C1 root-agg
evals (d2/d4/d8), rendering-noise + drift-curve measurement all running.
Smoke signal from the noise measurement (3 problems): rendering spread =
~49% of per-step state movement; 1-step predictor error only 1.9x that
floor — temp-variable-name randomness in step rendering may be a large
irreducible noise source (A2). Full 50-problem run in flight.

## 2026-08-25 (morning) — overnight board read out: solprob is the headline; C1/D1 dead; A1 knockout says the planner lives on the predicted manifold

All 12 finished Alex cells + all Grünau diagnostic lanes collected (fc = full_catalogue, 200 eps, metrics exact / answer / success at d1/2/4/8).

**Winner: `fresh-solprob`** (rollout_solution_prob>0 — energy candidate rollouts follow solution trajectories — plus the 1M-fresh-data protocol):
fc success 1.00/.95/.92/.815, answer .945/.835/.710/.590, exact .835/.653/.304/.172.
That is the best depth profile any cell has ever posted: d4 success .92 (baseline .85), d8 success .815 (baseline .74), and answer at d8 nearly doubles the field (.59 vs ~.42-.47 elsewhere). Depth "drop" in success terms is now 1.00→.815 over d1→d8.

Other cells: `fresh-horizon` (B2) solid second (answer d8 .42); `p16-freshdata` (1M data alone) does NOT fix depth (answer d4 .54) — fresh data helps but is not the mechanism; rollpred variants reconfirm the mid-depth exposure-bias gains (exact d4 ~.24-.31) without reaching solprob's success profile; `p16-late` still best exact-d8 (.206).

**Diagnostics (all on the prefix16 ckpt, eval-only):**
- A1 knockout (`--endpoints true` + oracle distance): WORSE than imagined endpoints (d2 success .695 vs .995 imagined; exact .000). With pred/enc norm ratio 2.35, encoded REAL states are off-manifold for the ruler — the planner's geometry is self-consistent on the predicted manifold even as it drifts from the encoder manifold. So "ground on real states at eval time" is not a fix; training-side grounding (rollpred/causal) is the right lever.
- A2 noise (50 problems): render spread LN-L1 .168 = 49% of per-step movement .339; 1-step pred err .312 = 1.86× the noise floor; drift k=1..8: .31→.61. Rendering noise is a real, sizable floor — temp-name canonicalization remains a candidate fix.
- C1 root-agg mean: worse at every depth (d4 exact .096 vs .169). Winner's curse is NOT the bottleneck. Dead.
- D1 aggregate endpoint/movement: both far worse than mean_prefix. Dead.
- Proposer lane unchanged: prior_propose answer stuck ~.55-.62 at d2 for every cell — proposer coverage is the independent second bottleneck, still needs the training-side off-path fix.

Still running: fresh-causal-rollpred (rtxpro6k), fresh-insert2 / fresh-insert3 / fresh-insert2-rollpred-k8 (a100), p16-rollpred-late.

**Planned next: combo round** — solprob + rollpred-k8 + late (+ horizon variant) on the fresh-data protocol; drift curve on a rollpred ckpt; then the proposer-coverage training fix.

## 2026-08-25 (midday) — search code certified; rendering-noise theory of the depth cliff; knockout lanes launched
- Symbolic-oracle ladder (2026-08-20) re-read: same beam/MPC code with a perfect scorer = 1.00 at d1/2/4/8 → search implementation exonerated. True-state + LN-L1 ruler ladder pooled: 1.00/.69/.62-.72/.64 — cliff at d2, flat after (no real recovery).
- Working theory (owner's): rendering noise (random temp names, spread ≈ 1/2 step) makes each action's future multimodal; mean-seeking regression blurs the predictor (drift), and at eval the same jitter contaminates every encoded-state distance. d1 survives because sibling comparisons share prefixes and have big margins; d≥2 compares across branches where nothing cancels.
- New eval-only knockout: `plan_flat --true-render-avg K` (commit 4cb9095) — average encodings of K re-renders of each REAL endpoint + goal, halving ruler noise at K=4. Lanes E3/E4 (trueend-ravg4-d2/d4, 200 eps) on gruenau2 GPUs 1/2. Also running: E1 endpoint-only aggregation trueend d2/d4, E2 width-256 trueend d2 + imagined endpoint-agg (gruenau1), trueend-d8 (g7).
- Predictions: noise theory → ravg4 lifts d2 well above .69; aggregation theory → endpoint-only lifts it; winner's-curse/width → expand256 drops it.

## 2026-08-25 (afternoon) — canonical rendering shipped as the default
- Root cause of render noise: reference renderer get_symbol() with symbol_method='rand' draws temp variable names from the process-global RNG. The renderer already ships a deterministic 'seq' mode; FaithfulEnv now forces it on its render copy (p2), so a trajectory always renders as the same text. Commit 73eec471211f39bd4834aeaae7b674dee312208d; regression tests in tests/test_canonical_rendering.py; TEXTJEPA_LEGACY_RENDER=1 reproduces the historical noisy rendering.
- Scope: all step rendering (training trajectories, counterfactuals, planning eval). Problem statements untouched — that randomness is problem identity, not label noise.
- OPS RULE: new training cells inherit canonical rendering automatically. Evaluating LEGACY-trained checkpoints must set TEXTJEPA_LEGACY_RENDER=1 (their encoders saw noisy renders). In-flight knockout lanes run from pre-canonical snapshots — unaffected.

## 2026-08-25 (afternoon-2) — canonical combo round submitted; NaN causal cell killed
- fresh-causal-rollpred had NaN grads FROM STEP 0 on Alex (grad-norm probe: encoder=nan for energy_prefix_rank at init; ran 14h on garbage) — killed (4097954). Causal predictor needs a numerics debug (suspect fp16/compile in the causal history path) before resubmission; local CPU tests pass.
- Submit-line recovery: fresh-solprob also carried latent_rollout_pred weight 1.0 — the overnight winner is solprob+rollpred, not solprob alone.
- New round (canonical rendering, snapshot 73eec47, warm-start prefix16, 1M×1 fresh data): canon-base, canon-solprob, canon-solprob-k8, canon-solprob-k8-late, canon-solprob-horizon, canon-solprob-insert2. Jobs 4101909-14, rtxpro6k/a100.
- p16-rollpred-late was CANCELLED at 10min on 08-24 (not a result); superseded by fresh-rollpred-late.
- Intermediate-signal watchers: 3 a40 jobs (4101976-78, alex_watch.sbatch) eval each canon cell's last.pt (fc d2+d4, 100 eps, canonical snapshot) every ~15min-cycle while training runs; results land in runs/<cell>/watch/t<HHMM>_d<D>.json. First curve points expected ~2-3h into training.

## 2026-08-25 (evening) — micro render-avg A/B: noise theory supported at d2
- Paired 30-problem trueend-oracle test, same episodes: d1 identical (1.000/.900 both arms — control passes). d2: 1-render .733 solve/.333 answer → 4-render-averaged .833/.500. Halving ruler noise recovers ~1/3 of the d1→d2 cliff; canonical rendering (zero noise) expected to recover more. 30 eps → ±9%; 200-ep confirmation (lane E3) in flight, micro d4 arm still running.
