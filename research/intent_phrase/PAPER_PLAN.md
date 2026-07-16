# Paper plan — controlled observed-intent language reasoning

_Revised 2026-07-14 for the causal-transformer rerun. This is the frozen
structure for completing the easy-domain evidence before moving domains.
Hierarchy is excluded from the paper-facing easy-domain section._

## Section-level claim

An action-conditioned causal latent world model can learn accurate counterfactual
reasoning transitions without reconstructing text. Its principal limitation
is non-symbolic action ordering rather than transition identity. Multi-step
latent-goal preference distillation closes much of this selection gap, while
the history-conditioned predictor supports recursive latent planning.

This is a controlled-mechanism section about flat latent dynamics, action
selection, and representation content. Hierarchical modeling is out of scope.

## Narrative order

1. **Controlled task and information-matched action interface.** Define state,
   observed intent action, rendered outcome, strict/+2 budgets, and the shared
   current-feasible-action menu. State explicitly that future feasible actions
   are unavailable to deployed planners.
2. **Action-conditioned latent dynamics.** Introduce direct next-state
   prediction with EMA targets and variance–covariance regularization. Establish
   counterfactual transition matching before discussing policy accuracy.
3. **Non-symbolic action selection.** Introduce multi-step latent-goal
   preference distillation as an environment-interaction teacher, distinct
   from exact symbolic ranking.
4. **Main comparison and cumulative build-up.** Compare the final flat JEPA to
   random, token-policy LM, sentence-policy LM, sentence+latent LM, and the
   annotated symbolic-preference reference. Directly below the baselines, show
   a cumulative JEPA ladder built under one identical protocol.
5. **Why each component matters.** Report matched one-component ablations and
   the GAR horizon/root-width and dense-rollout curves.
6. **What the representation contains.** Show transition, linguistic,
   task-progress, counterfactual, and effective-rank probes across the
   cumulative ladder.
7. **Implication.** State what the controlled graph establishes and motivate
   transfer to harder, less structured language reasoning without introducing
   an easy-domain hierarchy claim.

## Main table

All numeric rows use three independent training seeds, the same fixed
validation/test problem sets, and shuffled candidate menus. Columns are:
supervision tier, parameters/compute, strict success, +2 success,
counterfactual transition match, clean-history useful-action top-1, and
after-error useful-action top-1.

### Baseline block

1. Random feasible policy — already complete.
2. Autoregressive token intent policy — complete, three seeds.
3. Autoregressive sentence intent policy — complete, three seeds.
4. Sentence policy with auxiliary next-latent prediction, likelihood selection
   — complete, three seeds.
5. Sentence policy with auxiliary next-latent prediction, latent selection —
   complete, three seeds.

### Cumulative JEPA block

Use scientific row names and a single addition per row:

1. **Stabilized one-step latent dynamics:** direct action-conditioned
   transition, EMA target, variance–covariance regularization.
2. **+ observed-outcome embedding prediction.**
3. **+ recursive predicted-outcome consistency.**
4. **+ two-step latent-goal preference distillation** (`H=2, K=2`).
5. **+ dense four-step latent rollout**, only if it improves the common
   standard-distribution three-seed result. Otherwise place it in the negative
   ablation block.

The current reduced model already completes row 4 at
`.797±.008/.963±.008`; rows 1–3 and the common-protocol dense row require a
clean three-seed rerun. Do not construct this ladder from historical runs that
used different objective sets or problem-length distributions.

### Reference block

- Exact symbolic-preference latent model — complete, three seeds; visually
  separated and labeled annotated reference.
- Oracle policy — deterministic ceiling.
- Hierarchy is excluded from the paper table and narrative.

## Required three-seed ablations

Every ablation is made around the final flat objective set, changing one factor
only:

- remove latent transition prediction;
- remove observed-outcome prediction;
- remove predicted-outcome consistency;
- remove variance–covariance regularization;
- replace EMA target by online stop-gradient;
- replace direct dynamics by residual dynamics;
- add faithful observed-action displacement decoding;
- add scalar geometry-to-value regression;
- add terminal-distance monotonicity;
- add counterfactual transition/outcome prediction.

Report strict/+2, transition match, value-order correlation, clean-history and
after-error top-1, state scale, and effective rank. Historical one- or two-seed
screens remain provenance only.

## Hyperparameter curves

All plotted points are three-seed means with individual-seed markers:

1. **Preference teacher:** `H={1,2,4,8,16}` at fixed `K=2`; root alternatives
   `K={1,2,4,8}` at selected `H`; beam width shown as a teacher-quality/compute
   diagnostic, not silently mixed with student results.
2. **Dense rollout:** `N={0,1,2,4,8}` and discount
   `lambda={0.5,0.7,1.0}`. Report both horizon-specific prediction error and
   closed-loop success, since the lowest rollout error need not give the best
   policy.
3. **Regularization/stability:** EMA versus online stop-gradient versus fully
   online, crossed with none/VICReg/SIGReg and faithful action-displacement
   decoding off/on. If the full factorial is too large for the main text, put
   the complete table in the appendix but retain all runs.

## Representation analysis

Compare every cumulative JEPA stage and the final one-component ablations over
three seeds:

- state/action scale and effective rank;
- operation, entity/variable, numeric value, necessity, remaining-work, and
  terminal-answer probes;
- action recovery from latent displacement when LDAD is present;
- counterfactual next-state matching, RSA, and action-shuffle falsifier;
- linear value and useful-action ordering probes on clean and perturbed
  histories;
- temporal information curves over recursively predicted primitive states.

Do not claim emergent abstraction merely from lower prediction error. Relate
probe changes directly to counterfactual prediction or planning behavior.

## Figures and qualitative evidence

Main text figures:

1. Model/action-interface schematic.
2. Three-seed cumulative success ladder beside matched baselines.
3. GAR horizon/root-width curves.
4. Dense-rollout depth: prediction error and control success on aligned axes.
5. Representation probe heatmap across cumulative stages.

Appendix:

- full component factorials and all seed values;
- candidate-order and faithful action-menu corrections;
- 3–5 preselected matched episodes comparing JEPA, token LM, and sentence LM,
  sampled by fixed indices rather than outcome-based cherry-picking.

## Completion gates before switching the paper focus

1. Freeze the causal flat cumulative objective set on validation.
2. Run missing cumulative rows and every final one-component ablation with
   seeds `{0,1,2}`.
3. Run the reported hyperparameter curves and probes with the same three model
   seeds.
4. Regenerate all tables/figures from machine-readable artifacts, compile and
   visually inspect the Beamer deck, and seal the easy-domain final test.
5. Only then switch to the harder language domain under a separately frozen
   protocol.
