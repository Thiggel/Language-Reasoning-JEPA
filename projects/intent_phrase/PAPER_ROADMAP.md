# Roadmap to an ICLR-quality intent-phrase JEPA paper

## Target paper claim

An action-conditioned joint-embedding predictive model can learn
counterfactual reasoning dynamics without reconstructing language. In a
controlled observed-action setting, accurate latent dynamics are not enough:
the principal challenge is learning a non-symbolic action preference. A
properly calibrated latent-goal teacher and grounded transition objectives can
close that selection gap, while mechanistic probes reveal which task
information the predictive bottleneck retains.

The paper should not claim free-form language generation or discovered latent
actions. Stylized iGSM supplies controlled mechanism evidence; faithful iGSM
must supply the headline transfer result. A toy-only paper is unlikely to meet
the target standard.

## Stage 0 — seal the evidence contract

Before selecting another model:

- finish the repaired three-seed counterfactual-outcome row;
- regenerate every causal-matrix table from machine-readable artifacts;
- verify identical shuffled action menus, validation/test instances, budgets,
  and candidate information across JEPA and LM baselines;
- label symbolic preference, oracle terminal state, and future-feasible-action
  results as references rather than proposed supervision;
- keep the final stylized and faithful test sets sealed.

**Exit gate:** one manifest identifies every included checkpoint, seed,
configuration hash, exclusion, and evaluation split.

## Stage 1 — localize the causal-model gap

The highest-value next experiment asks:

> Does the causal J3 model fail because its dynamics are poorly optimized, or
> because the preference student fails to reproduce its latent-goal teacher?

First use existing checkpoints to report, by clean/perturbed history and
problem depth:

- counterfactual transition match and action-shuffle falsifier;
- teacher-versus-oracle and student-versus-teacher pair accuracy/top-1;
- value/energy calibration, margins, and selected-action regret;
- teacher-forced and recursive prediction drift;
- state variance, effective rank, and gradient contribution per objective.

Then run the smallest faithful causal-only screen needed by that audit:

- causal context window `{1,4,full}` at matched parameter count;
- a coarse predictor learning-rate/updates check around the current setting;
- preference-loss calibration only if teacher quality is healthy but student
  alignment is poor.

Do not mix new regularizers into this diagnostic.

**Decision rule:** fix dynamics/optimization if transition quality or drift is
the gate; fix preference calibration if the teacher is strong but the student
does not reproduce it; redesign the teacher if both student and teacher rank
actions poorly.

## Stage 2 — freeze the strongest non-symbolic recipe

Starting from the repaired causal reference, screen only the interventions
already supported individually:

- faithful observed-action displacement decoding;
- terminal-distance monotonicity;
- scalar next-state value calibration;
- observed counterfactual outcome prediction;
- recursive outcome consistency on/off.

Use a staged fractional factorial: single seed for clear failures, a second
seed for viable combinations, and three seeds only for the selected recipe
and its one-component removals. Include loss-scale and learning-rate checks so
an intervention is not rejected merely because its gradients are larger.

**Exit gate:** selected JEPA closes most of the matched token-policy gap on
validation, improves more than one seed, retains near-perfect transition
matching, and passes collapse/order/leakage controls.

## Stage 3 — decisive curves, not ornamental sweeps

After freezing the recipe:

- latent-goal teacher horizon `H={1,2,4,8,16}` and root alternatives
  `K={1,2,4,8}`;
- bounded continuation/beam compute at the selected horizon;
- dense rollout `N={0,1,2,4,8}` with only a small discount comparison around
  a viable nonzero depth;
- EMA versus online stop-gradient and properly tuned VICReg/SIGReg controls;
- direct versus residual prediction as a negative architectural ablation.

Plot individual seeds, teacher quality, student alignment, closed-loop
success, prediction drift, and compute. Do not imply monotonic improvement
where the measured curve peaks or reverses.

## Stage 4 — faithful iGSM transfer

Transfer the frozen recipe without retuning on the test set. Retrain JEPA,
token policy, sentence policy, and sentence-plus-latent policy with three
seeds, stable problem-specific shuffled action menus, and matched current
feasible-action access.

Report stylized-to-faithful degradation, graph-depth strata, after-error
recovery, calibration, and wall-clock/parameter cost. If the result does not
transfer, the paper becomes a controlled negative/mechanistic study unless a
new environment is approved; it must not silently headline the toy result.

**Exit gate:** a statistically stable faithful result and a clear advantage
on at least one paper-relevant axis—success, robustness, data efficiency,
counterfactual generalization, or planning compute.

## Stage 5 — representation science

For the cumulative recipe and matched ablations, measure:

- values, operations, entities, resolved variables, query relevance,
  remaining work, and terminal answer;
- action recovery from displacement when LDAD is active;
- counterfactual RSA and action-shuffle controls;
- clean-history versus after-error useful-action ordering;
- information decay along recursive latent rollouts;
- nonlinear/MDL and random-label selectivity controls.

Add causal interventions along robust probe directions. The paper should
connect representation changes to transition or planning changes; probe
accuracy alone is not evidence of useful abstraction.

## Stage 6 — final evaluation and paper package

- Seal one main table: matched baselines, cumulative JEPA build-up, final
  model, one-component removals, symbolic reference, and oracle.
- Use 3–5 fixed-index qualitative episodes comparing JEPA, token LM, and
  sentence LM, including at least one failure for each method.
- Report all seeds, uncertainty, parameters, training/planning compute,
  exclusions, and negative results.
- Build figures directly from artifacts and compile/visually inspect the
  paper deck.
- Run the final test once, after the configuration and analysis plan are
  frozen.

## What would make the paper genuinely strong

The strongest outcome is not merely beating the token LM by a point. It is a
coherent empirical explanation: JEPA learns excellent counterfactual dynamics;
selection is the bottleneck; a non-symbolic latent-goal mechanism fixes it;
the resulting states expose structured progress variables and generalize to
faithful, deeper reasoning. If faithful transfer or matched-baseline
performance remains weak, adding more toy ablations will not produce a
top-tier paper; the project will need a second approved observed-action
reasoning domain or a narrower mechanistic claim.
