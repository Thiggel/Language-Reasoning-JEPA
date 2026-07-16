# Wave 12 — causal-transformer paper matrix

_Started 2026-07-14; Stage 1 training/evaluation is complete except for the
repaired counterfactual-outcome row. Hierarchy is out of scope for the
paper-facing rerun._

## Frozen architectural change

Every newly trained flat JEPA uses a two-layer causal transformer predictor.
At transition `t`, its prediction attends to the complete teacher-forced
`(state, action)` prefix through `t`; open-loop evaluation recursively inserts
predicted states while retaining that history. Alternative-action predictions
are evaluated with independent causal prefixes, so candidates cannot attend to
one another. The primitive MLP and FiLM predictors remain loadable only for
historical checkpoint audits.

The two-layer width-256 predictor uses eight attention heads, a 4× feed-forward
width, direct next-state prediction by default, and no dropout. All conditions
use stable shuffled action menus and disable hierarchy (`macro_k=0`).

## Stage 1: cumulative ladder and matched ablations

Three seeds are running for each condition:

| family | conditions |
|---|---|
| cumulative | one-step dynamics; + observed outcome; + recursive outcome consistency; + H2/K2 latent-goal preference; + dense rollout depth 4 |
| removals | latent prediction; all outcome prediction; recursive outcome consistency; VICReg; EMA target |
| add-backs | residual prediction; faithful LDAD; scalar value distillation; goal monotonicity; counterfactual outcomes |

Each condition receives seeds 0/1/2. Seed-to-GPU assignment rotates across
conditions to avoid confounding a seed with GPU type. Every checkpoint receives
the standard probes, strict/+2 planning evaluation, representation plots, and
the GAR teacher audit where applicable.

### Stage-1 planning results available to date

These use 200 fixed validation problems per seed. They are paper-protocol
three-seed estimates, but model selection and the final test remain unsealed.

| condition | strict | +2 actions |
|---|---:|---:|
| one-step dynamics (J0) | .098 +/- .045 | .495 +/- .115 |
| + observed outcome (J1) | .123 +/- .076 | .500 +/- .109 |
| + recursive outcome consistency (J2) | .125 +/- .072 | .483 +/- .101 |
| + H2/K2 latent-goal preference (J3) | **.588 +/- .013** | **.845 +/- .043** |
| + dense rollout depth 4 | .510 +/- .106 | .812 +/- .062 |
| J3 without latent transition | .540 +/- .074 | .727 +/- .071 |
| J3 without observed outcome | .302 +/- .059 | .658 +/- .015 |
| J3 without recursive outcome consistency | .610 +/- .055 | .835 +/- .026 |
| J3 without VICReg | .532 +/- .039 | .752 +/- .072 |
| J3 with online stop-gradient target | .578 +/- .008 | .808 +/- .018 |
| residual rather than direct predictor | .358 +/- .137 | .682 +/- .095 |
| + faithful observed-action displacement decoding | **.632 +/- .040** | **.868 +/- .020** |
| + scalar value distillation | .602 +/- .041 | **.872 +/- .028** |
| + terminal-distance monotonicity | **.637 +/- .008** | .848 +/- .038 |

The causal rerun is materially weaker than the earlier reduced MLP reference
(.797/.963) and the matched token intent LM (.827/.978). The large J2-to-J3
jump confirms that non-symbolic selection supervision remains decisive. Dense
four-step rollout is not selected. Direct prediction is strongly preferred to
the residual parameterization. LDAD and monotonicity are promising add-backs,
but they are separate interventions and must not yet be combined or called the
final recipe.

The counterfactual-outcome cell initially failed before training because
advanced indexing swapped the time and batch dimensions in independent causal
prefix construction. The implementation and an exact-prefix regression test
are fixed; seeds 0/1/2 are being rerun from a self-contained cluster commit.

## Stage 2: selection-dependent curves

After Stage 1 selects the flat objective set on validation, run three seeds for
preference horizon `H={1,2,4,8,16}`, root alternatives `K={1,2,4,8}`, dense
rollout `N={0,1,2,4,8}`, and the minimal discount comparison needed at the
selected nonzero depth. Rerun the autoregressive token and sentence baselines
on the identical shuffled validation/test protocol. No hierarchy experiments,
figures, or claims enter this wave.

## Stage 3: stability factorial and matched baselines

Run the complete `EMA / online-stop-gradient / fully-online` ×
`none / VICReg / SIGReg` × `LDAD off / on` factorial over three seeds. The
autoregressive token policy, sentence policy, and sentence policy with an
auxiliary latent target are then retrained over three seeds on the same
shuffled-menu protocol. These models are causal baselines; hierarchy remains
disabled and absent from the report.

## Stage 4: faithful-iGSM confirmation

Only after validation freezes the flat stylized recipe, transfer that exact
causal objective set to faithful iGSM and retrain it and all three causal LM
baselines over seeds 0/1/2. Stable problem-specific action-menu shuffling is
mandatory. The faithful configuration is provisional until Stage 1 selects
between J3 and dense rollout; it must not be launched earlier.
