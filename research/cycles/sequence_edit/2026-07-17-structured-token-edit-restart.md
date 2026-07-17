# Cycle: structured token-edit JEPA restart

Status: implementation-valid CPU pilot; GPU plan awaiting controller admission

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
