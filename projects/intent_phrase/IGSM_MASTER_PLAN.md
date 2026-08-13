# iGSM master plan — one budget, one dataset, everything

_Frozen 2026-08-13. Every cell below trains at the SAME budget on the SAME
data unless it is explicitly a size ablation. Priority: finish iGSM 100%
before starting the other domains._

## Fixed protocol (do not vary except where stated)

- **Architecture / budget**: GPT2-small shape — d_model 768, 12 state
  layers, 12 heads, predictor 4 layers, batch 8. **157M params** as
  measured in our stack (their 124M + our action encoder, predictor and
  LDAD decoder). LM baselines matched: token LM d768/12 layers, sentence
  LM d768/12 state layers.
  Justification: Physics of LMs 2.1 uses 12x12x768 (124M) and reports
  DEPTH > WIDTH ("a 4-layer transformer, even with 1920 hidden dims,
  underperforms"); our old 4-layer config was their failure case.
- **Data**: FAITHFUL iGSM (their generator), **iGSM-hard** spec —
  train `max_op=21 max_edge=28 op_range=[3,21]`; OOD eval `op_range=[28,32]`
  (their exact bands). ID eval = held-out op<=21.
- **Eval**: success-vs-budget curve to **slack 16** with AUC (slack 4 is a
  wrong ruler at these lengths), planning depths {1,2,4,8,16}, LM loop
  counts {1,2,4,8,16} (never drawn as a depth curve).
- **Seeds**: 0-4 for every reported row.
- **Interfaces** (eval-time only, one training run serves all):
  feasible_menu (headline) | full_catalogue (no-oracle stress) |
  ldad_cycle (menu-free) | codebook_ground (menu-free, env-side grounding)
  | autonomous (decoder emits its own text, answer-only check).

## A. Main rows (5 seeds each)

| row | script | notes |
|---|---|---|
| JEPA LDAD (ours) | run_intent_horizon_energy_cell.sh mix4_aux025_nohorizon | + observed_action_ldad, action_codebook_k=256 |
| TD-JEPA | same, baseline_td_jepa | |
| GoalHead | same, baseline_goal_head | |
| token LM | run_intent_lm_screen_cell.sh token_lm | |
| sentence LM | " sentence_lm | |
| sentence LM + latent MSE | " sentence_lm_latent | two predeclared eval rules |
| token LM looped | " token_lm_rec | loop axis {1,2,4,8,16} |
| sentence LM looped | " sentence_lm_rec | |
| sentence+latent looped | " sentence_lm_latent_rec | |
| random / first-feasible / oracle | evaluator built-ins | no training |

LR: one compact screen per family at seed 0 ({1e-4,3e-4,1e-3,3e-3}), then
5 seeds at the winner. (Faithful winner at the OLD size was 1e-4; re-screen
at the new budget because LR interacts with depth/width.)

## B. Ablations (iGSM only, 5 seeds for plotted cells)

1. Predictor variants: causal-sequence, non-residual MLP (both already
   shown on-par at the old size — rerun at budget).
2. Geometry value loss: ranking weight x advantage-MSE weight grid.
3. Advantage horizon N x continuation beam width.
4. Counterfactual breadth K.
5. Dense recursive dynamics: rollout depth x discount.
6. Causal falsifiers: shuffled action/outcome alignment (must hurt),
   history masking, goal permutation, no transition loss, true vs
   predicted-state scoring.
7. Interface axis: the five interfaces above on the same checkpoints.
8. **Size ablation (the only budget exception)**: width {256,512,768} at
   fixed depth vs depth {4,8,12} at fixed width — replicates the paper's
   depth>width claim in our setting.

## C. Analysis / representation (on the 5-seed final checkpoints)

1. State-readout probes (resolvedness, feasibility) — JEPA vs token LM vs
   sentence LM at the same causal boundary, with shuffled-state and
   one-hot controls. (Method proven; rerun at budget, 5 seeds.)
2. Frozen-state sentence decoder: true vs imagined states, value/token/
   exact accuracy, prompt-only and one-step-from-true-prefix controls.
3. Linear numeric + categorical probes per the contract (remaining steps,
   progress, feasible-action count, op type, necessity).
4. Geometry diagnostics: effective rank, covariance spectrum, CKA between
   matched seeds, kNN purity; **t-SNE/PCA plots** of action-embedding
   clusters and state trajectories (illustration only, never evidence).
5. Counterfactual pairs: which interventions move the latent, action
   displacement composition, goal-distance calibration, rollout drift.
6. Causal controls: random init, label permutation, history masking,
   equal-dimensional random projections.

## D. Test-time scaling plots (headline figure)

- Success vs **planning depth** {1,2,4,8,16} at fixed budget, per band.
- Success vs **search width** (beam {1,4,8}) at fixed depth.
- Success vs **measured FLOPs / wall time** — JEPA depth curve against
  LM loop curve, matched compute (the LM rows contribute one cost point
  each per loop count).
- Success-vs-budget (slack) curves with AUC, ID and OOD bands.

## E. Order of execution

1. LR screens at budget (seed 0, all families) — RUNNING.
2. 5-seed mains, all rows, ID training.
3. Eval matrix: bands x interfaces x depths/loops.
4. Ablations (B), then analysis (C), then figures (D).
5. ONLY THEN the other three domains at the same budget.
