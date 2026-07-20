# Action-prior and hierarchy planning decision

## Decision

Do not promote current JEPA reranking or hierarchy. First factorize prompt-size and reasoning-length generalization using frozen checkpoints.

## Validity

All six recovery jobs completed with exit code zero, declared artifacts, and matching source-checkpoint hashes. The six original training-plus-evaluation jobs are process-invalid because evaluation crashed after training. Action-prior comparisons use three seeds and fixed problems. Hierarchy bottlenecks have one seed each. Oracle future-action results remain candidate-privileged.

## Evidence

- Prior-only exceeds JEPA-only at lengths 3, 6, and 9 for strict and +2 success.
- Top-2 and top-4 JEPA reranking give back prior-only gains despite top-4 necessary-action recall near .99 through length 9.
- Oracle future menus plus depth-4 rollout improve length-9 strict success from .061 to .439 and +2 from .750 to .994.
- Wide CEM hierarchy loses to first-feasible control for every bottleneck; code and prior-noise domains have identical success.
- The length evaluator changes prompt variables from 6–12 at length 9 to 24–36 at length 12, confounding the zero beyond nine actions.

## Outcome patterns for the next round

- Prompt-size effect only: length-9 success falls as variables increase, while fixed-prompt success changes little from length 9 to 12. Train across prompt sizes.
- Reasoning-length effect only: fixed-prompt success falls with length, while length-9 success is stable across prompt sizes. Train longer trajectories or repair recursive state dynamics.
- Both: report a two-dimensional generalization surface and vary both axes during training.
- Neither: audit positional interpolation and dataset/evaluator construction before new training.

## Human steering used

The round follows the request for harder length evaluation, deployable action priors, deep simulation, and controlled hierarchy. It also respects the instruction to distinguish candidate-privileged future menus and to avoid broad parameter sweeps before the final architecture is known.
