# Cycle: structured token-edit JEPA restart

Status: VICReg and LDAD screens complete; matched corruption/exposure control next

## Decision

Can a token-aligned recursive state make literal edits action-sensitive under
mask, random-replacement, removal, and curriculum corruption, and which of
EMA, EMA+VICReg, or EMA+VICReg+faithful text LDAD keeps that state healthy?

The user explicitly requested all four corruption regimes, the three
stabilizer conditions, pointer-based positions, deep recursive supervision,
goal-advantage distillation, and later multi-space hierarchy. The expired
overnight steering note still contributes two applicable constraints: remain
non-symbolic and use causal predictors. It no longer authorizes unattended
work by itself; the current user message explicitly authorizes submission.

## New interface

- Every current buffer is retained as contextual token latents rather than
  collapsed into one state before transition prediction.
- An action is `(operation, current token/gap pointer, content token)`. The
  predictor gathers the neighboring token states, so prefix insertion shifts
  the pointer but not the local action representation. Textual decimal
  positions remain only in the externally observed phrase decoded by LDAD.
- Delete/insert/replace constructs the exact token-latent scaffold. A
  zero-dropout bidirectional spatial Transformer predicts contextual next-token
  latents. Rollout is causal across edit time and recursively feeds predicted
  token states through this same interface.
- EMA targets are permanently in evaluation mode. VICReg acts on online
  states. Faithful text LDAD sees only the online displacement and reconstructs
  the complete externally observed action phrase.
- GAR distills `d(s,g)-d(s_H,g)` into `V(s,a)` without giving the head `g`.
  `H=1` is immediate improvement; `H=4` follows the observed repair
  continuation and is terminal-privileged, not proof of a globally optimal
  continuation.

## Falsifiable pilot

The matched one-seed matrix crosses four corruptions with three stabilizers.
Two additional curriculum cells compare H=1 and H=4 GAR. All use 2,000 unique
oracle-denoising trajectories, three epochs, token rollout supervision through
depth four, no counterfactual alternatives, and no hierarchy. Corruption
families are placed on different backends for elapsed-time parallelism; the
three stabilizers within each family remain on the same backend. Cross-family
differences therefore require a later same-backend confirmation.

Primary gates:

1. shuffled/matched one-step error ratio at least 1.05;
2. prediction beats the no-change baseline;
3. recursive error reported at depths 1, 2, and 4 with no dormant predictor;
4. finite optimization, state effective-rank loss at most 10% relative to the
   best healthy condition, and exact EMA-eval/unit fixtures;
5. LDAD is retained only if it improves transition/recursive metrics, not only
   action-token accuracy.

No hierarchy, K/data expansion, or MPC result is admissible until a primitive
cell passes all causal gates. The token-to-span/phrase/sentence hierarchy and
factored beam/CEM planner are implemented only after this selection because
otherwise they would build on a rejected transition model.

## Stabilizer coefficient screen

The initial inherited coefficients were VICReg `1.0` and observed-action LDAD
`0.2`.  VICReg uses standard-deviation target `1.0`, covariance multiplier
`0.04`, and action-variance multiplier `0.1`; EMA momentum is cosine-scheduled
from `0.99` to `0.999`.  These were not calibrated for the new token-aligned
objective.

The coarse screen keeps curriculum corruption, data, seed, optimizer, EMA,
zero dropout, and all dynamics losses fixed.  First compare EMA alone against
VICReg weights `{0.02, 0.1, 0.5, 1.0}`.  Then compare LDAD weights
`{1, 10, 20}` at the healthiest VICReg coefficient, retaining the no-LDAD
cell as the matched control.  Add a `1e-4` learning-rate cross-check for any
LDAD coefficient whose gradients are finite but materially alter the total
loss scale.  This staged screen avoids spending a full Cartesian product on
collapsed or action-blind regions while still testing the paper-aligned LDAD
range requested by the user.

The Grete pre-training failures were caused by module-level visualization
imports in the bundled iGSM generator.  Matplotlib imports are now lazy and
confined to drawing methods; a headless import fixture and 13 targeted
faithful-edit/LDAD tests pass.

## Implementation validation

- Five new structured-recipe tests pass, including exact corruption recovery,
  EMA evaluation mode, prefix-shift invariance of pointer actions, structured
  forward execution, recursive shape equality, and finite GAR loss.
- The existing faithful edit suite passes 11/11 and the legacy edit suite
  passes 3/3.
- A tiny end-to-end CPU train/evaluate/checkpoint/audit run completed with
  finite token one-step and recursive losses. Its two-example metrics are a
  process smoke only, not scientific evidence.

## LDAD result

All five faithful text LDAD cells completed from commit
`dda1305ff9b18a5a01daf36e3a1b0683eda8319f`. LDAD weights 10 and 20 sharply
improve one-step and recursive mixed-corruption error and preserve effective
rank, but every shuffled/matched ratio remains between 1.006 and 1.011. The
action decoder reaches about 82% token accuracy without forcing the forward
predictor to use the supplied action. LDAD 20 has the best raw errors and a
peak pre-clipping gradient norm of 241.7, so it is retained only for a narrow
`1e-4` learning-rate falsifier. The decision-grade report is
[`2026-07-17-structured-edit-ldad-screen`](../../reports/sequence_edit/2026-07-17-structured-edit-ldad-screen/REPORT.md).

## Corruption and exposure audit

The nominal four-way comparison was not information/protocol matched.
Mask-only, replacement-only, removal-only, and mixed datasets ignored
`fresh_per_epoch: true` and repeated the same corruption trajectories, while
curriculum incorporated epoch into its random seed. Moreover, mask checkpoints
were evaluated on mask corruption and curriculum checkpoints on mixed
corruption. In a 64-example diagnostic, mask EMA obtains shuffled/matched
`1.163` on mask but only `1.003` on mixed edits; curriculum EMA obtains
`1.013` on mask and `1.014` on mixed. Thus mask action sensitivity is
distribution-specific rather than evidence of transferable mixed editing.

The dataset now has an explicit, tested `fresh_per_epoch` option, checkpoint
construction passes it through, and the model audit accepts an explicit
evaluation corruption override. Seventeen relevant structured/faithful edit
tests pass. These changes do not alter or retroactively relabel old snapshots.

## Next falsifiable decision

Ask whether transferable mixed-edit action sensitivity is controlled by
corruption family or fresh trajectory exposure. Compare fixed and fresh mixed
EMA, fresh mask EMA, and the existing fresh curriculum protocol under common
mask and mixed evaluation. Include one LDAD-20 `1e-4` optimization control
because its loss scale materially changed gradients. Advance only a condition
that reaches shuffled/matched `1.05` on mixed evaluation, beats persistence,
retains rank, and remains recursively stable. Counterfactual density,
hierarchy, GAR, MPC, and structured action-field LDAD remain gated.

## Corruption exposure result and metric correction

All five wave-3 jobs completed. Fixed and fresh mixed EMA are effectively
tied, mask sensitivity remains mask-specific, and lower-rate LDAD 20 remains
pooled-action-insensitive while reaching a peak gradient norm of 951. A direct
token-level shuffled-action smoke audit changes the conclusion: fresh mixed
EMA scores 1.012 after pooling but 3.018 across token-aligned targets. The
pooled causal gate is therefore invalid for this interface. Freeze training
and re-audit all five checkpoints and four corruptions before selecting or
scaling a recipe.

## Token causal re-audit

All five frozen checkpoints completed a 256-example, four-regime re-audit. On
mixed evaluation, token-level shuffled/matched ratios are 3.12--3.30 for EMA
recipes and 1.83 for lower-rate LDAD. Fixed versus fresh mixed training remains
indistinguishable. Single-operation global ratios remain diluted, so the next
frozen audit separates operation, current-buffer pointer, and content and
scores a radius-two local window. No scale round is admitted until this
component gate is resolved.

## Component-local result

Five 256-example frozen audits completed. Fixed mixed EMA gives local
operation/pointer/content shuffled-to-matched ratios 2.69/2.63/1.39. Fresh
mixed and curriculum are effectively tied with it. Lower-rate LDAD improves
recursive token error from about 0.308 to 0.214 but weakens the three causal
ratios to 1.58/1.61/1.25. Select fixed mixed EMA for the unique-data/exposure
and capacity screen. The token-aligned forward path does not consume existing
pooled counterfactual fields, so no K ablation is admissible until structured
alternative transitions and targets are implemented.

## Data and capacity result

All five wave-6 jobs completed. Exposure-matched 2k×3 versus 6k×1 and 6k×3
versus 18k×1 pairs are effectively identical. Increasing total presentations
from 6k to 18k improves matched token error from about 0.185 to 0.095 and
recursive token error from about 0.309 to 0.182. The d512 diagnostic improves
error but used batch four and twice as many optimizer steps; it is not an
isolated width result. LDAD 20 is worse after only 6k presentations, so its
earlier advantage depends on adequate optimization. Run frozen component-local
audits before extending scale.

## Scale component audit and structured alternatives

All five frozen component audits completed. The exposure-matched 6k×3 and
18k×1 cells again agree, including operation/pointer/content local ratios near
2.92/3.24/1.28. D512 is strongest at 3.33/3.45/1.48 but remains confounded by
twice the optimizer steps. Exact token-aligned mechanical alternative actions
and EMA outcome targets are now implemented without goal, preference, or
quality labels. An 18-test process suite and a CPU training smoke pass. Admit a
common-batch K={0,1,4,8} pilot before any counterfactual scale claim.

## Structured counterfactual breadth result

All five wave-8 cells completed. At common 2k states, batch two, one epoch, and
1,000 optimizer steps, K=1 improves matched token error 0.195→0.177 and the
global token causal ratio 3.108→3.365, but recursive token error worsens
0.322→0.330. K=4 and K=8 at weight one are numerically identical to K=1;
K=4 weight 0.25 is intermediate. Additional alternatives beyond one are not
justified. Run frozen local-component audits before confirming K=1 at the
18k-presentation anchor.
