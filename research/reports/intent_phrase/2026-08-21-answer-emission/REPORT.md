# Answer-emission success criterion (2026-08-21)

Round: `runs/autonomy/intent_phrase/2026-08-21-answer-emission-v1/`
(snapshot `runs/autonomy/_code/5d1e8b8`, checkpoint
`_ckpt_snapshots/ecf16-s0-best-2026-08-20.pt`, 200 episodes/cell, depth 1,
`--no-beam-diagnostics`, fp32, faithful iGSM).

## Why

Until now the JEPA planner was scored by the environment's own solved bit
(`query in resolved_set`, with the environment doing all arithmetic), while
the token-LM baseline had to generate the full trace and the answer itself.
Different tasks, incomparable numbers, and an obvious reviewer attack.
Owner decision: success = the system EXPLICITLY EMITS the answer, the same
rule for every system.

## The criterion, exactly as implemented

`success_answer` (in `src/textjepa/planning/flat_search.py`) requires ALL of:

1. the environment is solved within the runaway cap (the executed intents
   actually resolve the query — same as before);
2. at the step that resolves the query, the MODEL's token head greedily
   generates the final outcome sentence itself, from the token history that
   ends with the solving intent phrase.  The environment's rendering of that
   outcome is NOT yet in the context, so the model must do the final
   arithmetic, not copy it;
3. that generation terminates by the model's own choice — a sentence-final
   '.' token within the 96-token cap, not cap exhaustion;
4. the final integer of the emitted sentence equals the true answer
   (mod-23 arithmetic; the training traces' final sentence carries the
   answer in exactly this form, so the emission is in-distribution — the
   checkpoint was trained with `lm_loss_on=all_solution`).

The env-side outcome sentence is appended to the context afterwards (except
in `autonomous` mode, where the model's own sentence is kept), so planning
itself is unchanged and `success_env` (old criterion) is reported from the
same episodes.  No symbolic readout of env state feeds the model's answer.

The SAME rule is applied to the random and first-feasible reference rows:
they pick actions randomly/greedily but get the same emitter (the
checkpoint's token head) at their solving step.  This is the GENEROUS
reference — a random policy without a language model emits nothing and
scores exactly 0 by definition.

The LM baseline was already scored this way (`plan_lm.py`,
`gen_outcome=model`, `invalid_policy=fail`): the LM writes every outcome
sentence itself and success requires its own final sentence to state the
true answer.  We added requirement (3) (proper termination) there too.  A
degenerate/random LM fails at its first unparseable intent (episode ends,
`fail` policy) and can therefore not score; random under the LM protocol
is 0.

## Budget re-scoring rule under the criterion

`scripts/rescore_budget.py`: success_answer at cap c =
`solved_at <= ceil(c * necessary)` AND `answer_correct is True`.
This is exact, not an approximation: the answer is emitted AT the solving
step, so whenever `solved_at` is inside cap c the emission also happened
inside cap c; an episode that only hit the runaway cap has
`solved_at = None` and never counts at any cap, so "terminated by the
model's own progress, not by budget exhaustion" is implied by the rule.

## Headline table: success_env vs success_answer

ID = training distribution (max_op 15, max_edge 20, op 3..15).
nec3-5 = the harder band (3..5 necessary steps => larger useless-action
fraction).  All cells depth 1, 200 episodes, seed 0, split seed 2.

| cell | arm | env@1.0 | ans@1.0 | env@1.25 | ans@1.25 | env@4.0 | ans@4.0 |
|---|---|---|---|---|---|---|---|
| full_catalogue ID | model | .740 | .720 | .855 | .830 | .995 | .935 |
| full_catalogue ID | random+emitter | .000 | .000 | .010 | .010 | .595 | .285 |
| full_catalogue ID | first+emitter | .000 | .000 | .015 | .015 | .430 | .185 |
| full_catalogue nec3-5 | model | .780 | .780 | .825 | .825 | 1.000 | .990 |
| full_catalogue nec3-5 | random+emitter | .005 | .005 | .055 | .055 | .455 | .315 |
| full_catalogue nec3-5 | first+emitter | .000 | .000 | .030 | .030 | .415 | .240 |
| prior_propose ID | model | .760 | .740 | .810 | .785 | .855 | .825 |
| prior_propose ID | random+emitter | .000 | .000 | .010 | .010 | .595 | .300 |
| prior_propose nec3-5 | model | (cell still running 2026-08-20 night; fill from ood_prior_d1/plan.json) |
| prior_propose nec3-5 | random+emitter | (same) |

Token-LM baseline under the same idea (`gen_outcome=model`, existing runs,
before the termination tightening; 200 episodes each):
faithful-hard21 env .830 -> answer .815; iGSM-med env .810 -> answer .795.

## Where the delta is large (where the old numbers were inflated)

- The PLANNER loses little: on solved episodes its emitted answer is right
  94% (ID) / 99% (nec3-5).  At tight budgets (cap 1.0/1.25) the headline
  moves by <= .025; at cap 4.0 ID it moves .995 -> .935 (long episodes,
  longer contexts, more arithmetic to carry).
- The REFERENCE rows lose a lot at generous budgets: random at cap 4.0
  drops .595 -> .285 (ID) and .455 -> .315 (nec3-5) — of random's env-solved
  episodes only ~52-69% get the answer emitted correctly, because the final
  arithmetic must be computed over a long, distractor-filled context.  So
  the OLD criterion at the OLD default cap inflated the reference policies
  most; the planner-vs-random margin is now wider, and random is exactly 0
  at cap 1.0 as the owner predicted.
- Nothing about the planner's ranking or search changed; `success_env`
  from these runs reproduces the 2026-08-20 numbers (.740/.855/.995 ID
  full_catalogue) exactly.

## Fine print

- 1 of 199 solved ID planner episodes failed on the termination
  requirement alone (cap-exhausted emission); 12 emitted a wrong value.
- The emitter for the reference rows is candidate-generous and should be
  labelled "random actions + our emitter" in the paper; a bare random
  policy emits nothing and is 0 at every budget.
- Rescoring caveat: files written before this round lack `answer_correct`
  and report answer columns as `--`.
