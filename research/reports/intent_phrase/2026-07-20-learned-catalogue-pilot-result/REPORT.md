# Why the learned action catalogue selected invalid intents

## The one-sentence answer

The two valid pilot jobs show that the non-oracle planner usually retained a feasible action in its top-four proposal set but almost always selected an infeasible root, exposing a mismatch between feasible-only prior training and full-catalogue deployment rather than evidence about whether deeper JEPA simulation helps.

## First, the idea in everyday language

Imagine solving a worksheet by choosing the next instruction from a list. Some instructions are possible now, some become possible later, and some are irrelevant. Our earlier planner was unrealistically handed the list of possible future instructions. The new planner instead reads a fixed list from the prompt and uses learned scores to decide what may be available.

The first pilot behaved like a person who keeps four reasonable cards in hand but then picks a forbidden card when asked for the final answer. That sounds contradictory until we inspect training: the final preference score learned only from comparisons among already allowed cards. Scores for forbidden cards were never pushed down, yet those scores entered the full competition at test time. This report records that failure and the narrow correction; it does not treat the failed controller as evidence against latent simulation.

## Why this question matters

A deployable planner cannot query a hidden symbolic engine for actions that become feasible after imagined steps. If a learned action catalogue works, prior and JEPA planning can be compared without privileged future menus. This is necessary for the paper claim that more learned simulation—not more oracle information—improves control. It also separates two errors: failing to propose a useful action and choosing badly after useful actions have been proposed.

## What we tested

We started from the same seed-zero width-256 causal J3 checkpoint and froze its state encoder, action encoder, causal predictor, value function, and target encoders. Only a learned availability head and behavioral-prior head were adapted for five epochs on 40,000 generated problems. The two learning rates were 3e-4 and 1e-3.

Evaluation used exact necessary-action lengths six and nine, zero or two excess actions, top four catalogue roots, per-root beam width four, depths one, two, and four, and 30 deterministic episodes per cell. Availability weights .5, 1, and 2 were crossed with prior-only and JEPA trajectory scoring. Future symbolic feasible menus, an oracle goal, and oracle execution fallback were disabled. The environment was consulted only after the selected root was committed, so an invalid choice was counted as failure.

## What a fair comparison means here

The two learning rates used the same source checkpoint, data sizes, seed, frozen modules, action catalogue, evaluation instances, and planning budget. Prior-only and JEPA scoring received identical proposed trajectory banks. The action catalogue is prompt-derived and therefore available to both methods, but the behavioral demonstrations are explicit policy supervision and are not an action-free JEPA signal.

Both jobs completed and produced all expected artifacts with finite metrics and matching source-checkpoint hashes. There are only 30 episodes per cell and one training seed, so tiny differences are not decision-grade. The near-total invalid-selection failure is large enough to be a mechanism gate, not a leaderboard comparison.

## What happened

| Observation at exact length nine, strict budget | Result |
|---|---:|
| Evaluation cells across both jobs | 36 |
| Cells with zero success | 35/36 |
| Typical selected-action invalid rate | 1.00 |
| Best cell: success / invalid rate | .033 / .933 |
| Top-four feasible precision across cells | .54–.75 |
| Top-four necessary-action recall across cells | .69–.84 |

Length six was similarly unhealthy: only a few cells reached .033 strict success and invalid rates remained .93–1.00. Changing the availability coefficient did not repair selection. Therefore neither a favorable JEPA depth curve nor a fair prior-versus-JEPA comparison exists in this round.

## The intuitive picture

![The prompt action catalogue produces a top-four set that often includes feasible and necessary actions, but the final root is almost always infeasible because infeasible prior logits did not compete during training. The correction trains the prior across the complete catalogue.](catalogue_failure.svg)

The important distinction is between coverage and ranking. Coverage was imperfect but nonzero and often useful. The catastrophic event occurred at selection: an action outside the true feasible set won. This directs the next experiment toward the prior loss rather than wider beams, deeper rollouts, or larger models.

## The technical details

The availability head receives a latent state and each encoded intent phrase, and is trained with weighted binary cross entropy over all real prompt actions. The behavioral-prior head receives the same kind of pair and is trained with cross entropy toward the demonstrated next action. In the completed pilot, the prior loss mask was `step_mask AND candidate_mask AND action_feasible`. At planning time, the score was applied to the complete learned catalogue. Consequently, logits for infeasible actions had no direct comparative prior gradient.

The correction introduces `action_prior_candidate_scope` with values `feasible` and `catalogue`. `feasible` remains the default so historical experiments keep their exact semantics. The learned-catalogue experiment selects `catalogue`, making the loss mask `step_mask AND candidate_mask`; the demonstrated feasible action competes against every genuine action in the prompt. Padding is still excluded. The separate availability loss is retained for interpretability and future-state support scoring.

This correction is deliberately behavioral. It teaches both immediate feasibility and demonstrated usefulness through the prior denominator. It does not claim that feasibility emerged from JEPA dynamics. The checkpoint eligibility gate now requires catalogue-wide prior supervision for learned-catalogue planning, preventing accidental evaluation of older incompatible heads.

The implementation followed a regression-first check. Before implementation, the model rejected the new option and the checkpoint gate accepted feasible-only prior training. After implementation, the new test confirms that infeasible actions are included in the catalogue loss and that raising an infeasible logit increases loss. Legacy feasible-only masking still passes its original test. The full suite reports 120 passed. A checkpoint-initialized CPU smoke trained the intended 0.28 million head parameters with finite losses and completed learned-catalogue evaluation.

Raw artifacts are under `runs/autonomy/intent_phrase/2026-07-20-intent-learned-catalogue-planner-pilot-v1/`. The immutable pilot commit is `1e6fc9d5d268ccbaed18e138c6a5b5333a4fd62f`; the correction commit begins `eb5a599`.

## What we can conclude

Directly observed: useful actions were often present in the proposed root set, but the selected action was almost always invalid. The jobs and artifact contracts were valid. Availability-weight changes did not make the system usable.

Supported inference: the completed round fails the proposal-validity gate, so its JEPA-versus-prior differences should not be interpreted. Code inspection reveals a precise train/deployment support mismatch capable of producing the observation: infeasible prior scores were unconstrained in training but active in deployment. This is a stronger next target than additional depth or beam tuning.

## What we cannot conclude

We cannot yet conclude that catalogue-wide prior training will solve planning, that the availability head is calibrated on predicted future states, or that JEPA improves upon a healthy prior as depth grows. We also cannot infer robustness across seeds, widths, harder lengths, or faithful language. The fixed prompt catalogue assumes all possible intent phrases are enumerable from the prompt; environments requiring open-ended action generation need another mechanism. Finally, because the prior uses demonstrations, success would support a supervised proposal-plus-JEPA system, not a purely self-supervised planner.

## What happens next

The smallest faithful recovery trains only the same two heads at learning rates 3e-4, 1e-3, and 3e-3. It fixes availability weight at 1 because the earlier coefficient cross-check did not change the failure. It evaluates the same top-four, beam-four, depth-one/two/four protocol. The primary gate is length-nine selected invalid-action rate below .25, supported by improved catalogue-wide top-one accuracy. Only if this passes will we inspect whether JEPA gains over the prior with depth. Failure at all three rates redirects work toward a ranking-aware availability objective or a revised proposal factorization, not a dense scale sweep.

## Words used in this report

- **Action catalogue:** Every intent phrase explicitly available from the problem prompt, whether executable now or later.
- **Behavioral prior:** A supervised score trained to prefer the demonstrated next intent.
- **Feasible action:** An intent whose prerequisites are currently satisfied.
- **JEPA:** A model that predicts future representations rather than reconstructing every output token.
- **Invalid action rate:** Fraction of episodes in which the selected root cannot be executed in the true environment.
- **Necessary-action recall:** Fraction measuring whether proposal roots include actions on a valid solution path.
- **Oracle menu:** A privileged list of feasible future actions obtained from hidden symbolic state.

## Questions for you

- If catalogue-wide supervision passes the validity gate, should the next replication prioritize three additional seeds at width 256 or an early width-512 capacity check?
- For the final paper claim, is explicit demonstrated-action supervision acceptable as the proposal mechanism if JEPA alone performs the consequence reranking?

