# 2026-08-14 — Faithful-iGSM evaluation protocol fix (bands, budgets, masking)

Trigger: random policy scored .81 on the "hard OOD" full_catalogue band —
an inverted, obviously broken ruler. Full audit of the faithful-iGSM eval
path; three independent defects found and fixed; new instrument verified.
Code: snapshot `8acd62c` (protocol e2ca466 + mask fix 8acd62c).

## Defect 1 — banding on n_op against the generator cap starves distractors

Old bands selected problems by total operation count (n_op) with the OOD
band (op 28-32) butting against the generator cap max_op=32. Measured
consequence: ~75% of the action catalogue was necessary, leaving ~5
distractors. Under any menu/catalogue interface a wasted pick cannot be
repeated (it either resolves or is masked), so total waste is bounded by
the distractor count — with a fixed slack of 4, random could hardly fail
(.81). Independent simulation reproduced the recorded numbers
(.633 ID / .800 OOD vs recorded .605/.81). **Fix**: band on SOLUTION
LENGTH via new `necessary_range` (ancestors of the query node) with
generator caps far above the band, so distractors are plentiful:
ID nec 8-15 @ max_op=32/max_edge=40/op_range=[3,32];
OOD nec 20-28 @ max_op=48/max_edge=64/op_range=[3,48]
(max_op=32 measured to never yield nec≥20). `gen_problem` now REJECTS
off-band draws and raises after 200 attempts instead of silently
returning an off-band problem. Note: eval must also override
`eval_op_range` to the caps, else the training range [3,21] makes the
OOD band unsamplable (crashes, correctly).

## Defect 2 — fixed absolute slack

A fixed slack (e.g. 4) means long problems get proportionally tighter
budgets — one source of the ID/OOD inversion. **Fix**: proportional
`slack_frac`; attempt budget = necessary + slack + ceil(frac×necessary).
Paper instrument = success vs attempt budget in MULTIPLES of necessary
(blind search needs ~(catalogue/menu-width)×necessary ≈ 5×; a
feasibility-aware planner should approach 1×).

## Defect 3 — permanent attempted-mask made episodes unsolvable

The planner masked every attempted action forever; a necessary action
tried before its dependencies resolve was lost → full_catalogue success
0.000 for ALL policies including random. **Fix**: mask is scoped to the
current state — holds resolved actions plus invalid attempts since the
last progress, and resets whenever a feasible step lands. Regression
test added (`test_masked_catalogue_episodes_stay_solvable`).

## Structural finding: feasible_menu is unsalvageable as a main result

Faithful-iGSM feasible menus are intrinsically ~2-3.7 wide with 41-78%
necessary options, and wasted picks are unrepeatable → random succeeds
.83/.73 (banded probe) up to .98 (legacy slackcurves). No cushion choice
fixes this. **Decision (owner)**: feasible_menu → appendix control; main
results use menu-free interfaces (full_catalogue, codebook_ground,
ldad_cycle, autonomous).

## Verified instrument + hardened model finding

Smoke (`runs/autonomy/intent_phrase/2026-08-14-protocol-fix-smoke-v1/`,
hard-ldad-lr3e4 ckpt, 100 eps, budget 5× necessary, full_catalogue):

| band | planner | random | first-cand | planner invalid-rate |
|---|---|---|---|---|
| ID nec 8-15 | .530 | .700 | .510 | .793 |
| OOD nec 20-28 | .140 | .480 | .100 | .882 |

Ordering is sane (OOD harder for everyone), mid-range, discriminative.
The menu-trained Energy sits BELOW random menu-free: it has no
feasibility signal, because training never required rejecting infeasible
actions. This is the headroom the ldad_cycle/codebook_ground port (#28)
must fill (stylized-iGSM feasibility AUC .94). Paper narrative:
menu-trained energy fails menu-free → cycle-consistency recovers
feasibility → autonomous operation.

## Consequences / open items

- 12 pre-fix full_catalogue OOD JSONs are VOID (snapshot predated
  interface fix; invalid_action_rate exactly 0.000) — re-run from
  ≥8acd62c (#29).
- Budget multiples for the paper curve to be fixed (~{1,1.5,2,3,5}×) (#30).
- Checkpoints were trained with op_range [3,21]; eval bands now exceed
  that — decide whether mains need retraining on the wider range.
