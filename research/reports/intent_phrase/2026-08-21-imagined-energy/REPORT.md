# Imagined-energy training: new depth-1 bests, depth collapse diagnosed to its second root cause

Date: 2026-08-21 · faithful iGSM-med · commits 03bb351/0706638 (objective),
5d1e8b8 (rollout label fix, separate round) · run dirs
`runs/autonomy/intent_phrase/2026-08-21-imagined-energy-v1/` and
`2026-08-20-rollout-solution-v1/` (each with HANDOFF.md and NOTES.md).

## The two root causes of the depth collapse (established this round)

The true-oracle ladder (`2026-08-21-true-oracle-ladder/`) proved the search
machinery loses nothing; the learned ruler carries the whole collapse. Two
independent training defects explain *why* the ruler fails off the root:

**A. Train/test state gap.** The energy head only ever scored real encoded
states in training; deep search scores predictor-imagined states. The
decomposition probe (`probe_flat_energy_progress.py --reencode`, new) shows
this is not imagination error: even *truly re-encoded* off-reference states
score at chance (progress AUC d0→d4: imagined .832/.698/.645/.609, re-encoded
.832/.653/.535/**.494**; legality d4 .951/.707). Any state off the reference
trace is out-of-distribution for head and encoder alike.

**B. Worthless positives at depth (the deeper defect).** The training rollouts
that define "the continuation that actually occurred" at imagined depth stepped
`feasible[rng.randrange(len(feasible))]` — uniformly at random among legal
actions. So beyond depth 0 the ranking positive was itself a random legal
action, and ranking it against counterfactuals teaches legal-vs-illegal and
nothing else — exactly the probe signature (legality ~.97, goodness ~chance).
The counterfactual negatives were always present; the positive carried no
quality information. This also retro-explains why every earlier
energy-objective variant plateaued the same way.

Both fixes are self-supervised and stay inside the normative contract:

- `energy_imagined_rank` (fix A): roll the predictor h steps from a real
  prefix under the observed actions; at each imagined state rank the observed
  continuation below kcat sampled catalogue counterfactuals with the planner's
  exact energy call. Label = which continuation occurred. Config-gated,
  default 0; no junk-appending (length-matched by construction).
- `rollout_solution_prob` (fix B): rollout steps continue along the reference
  solution with probability p — the same rule that already generates the
  factual trajectories (`rng.choice(nec)`); the rollout walk was simply
  inconsistent with it. p=0 verified bit-identical to the old behaviour.
  Report p ∈ {0, 0.5, 1.0}; never p=1.0 alone (the probe walks a *random*
  feasible trajectory, so it tests off-path generalisation).

## Fine-tune with fix A alone (`imagined-ecf16-ft-s0`, COMPLETED)

Warm-start from `ecf16-s0-best-2026-08-20.pt`, ecf16 recipe + imagined term
(weight 8, depth 4, kcat 8), lr 3e-5, 4 epochs on a shared H100. Its own
training accuracy on the new task reached .997.

Planning (200 episodes, ID, self-run final eval; success at cap 1.0 / 1.25):

| interface | d1 | d2 | d4 |
|---|---|---|---|
| full_catalogue | **.820 / .900** | .525 / .660 | .285 / .400 |
| prior_propose | **.845 / .890** | .570 / .595 | .440 / .460 |

Depth-1 numbers are the **best recorded to date** (previous bests .740–.790 at
cap 1.0). Depth is still non-monotone.

Mechanism probe after training (progress AUC by depth, vs pre-training
baseline in parentheses): d0 .809 (.842), d1 .667 (.695), d2 .653 (.615), d4
**.650** (.606). The depth-wise *decay* flattened (−.23 → −.16 d0→d4) but the
off-root level is still ~.65 — because fix A alone trains on imagined states
whose positives are still random-feasible (defect B), so it partly re-learns
legality. The two fixes are complementary; neither suffices alone.

## In flight

- `combo-imgsol-s0` (gruenau11:3): **both fixes together** — imagined-rank
  weight 8 + rollout_solution_prob 0.5, warm-start ecf16. The decisive depth
  bet; its job.sh runs the AUC probe and d1/d2/d4 plans itself.
- `solp100-s0` / `solp050-s0` (fix B alone, from-scratch-recipe fine-tunes,
  epoch 9+) plus a step-matched control probe at `matched_step11k/`.

Decision rule (recorded in the round HANDOFF): success = progress AUC at d2–d4
materially above ~.65 AND d2/d4 planning no longer strictly below d1 at strict
caps, with invalid-action rate as the secondary metric.

## Honest ceiling

iGSM has no dead ends, so even a perfect ruler gains nothing from depth
(symbolic oracle: 1.000 at every depth *including depth 1*). The achievable
claim in this environment is "depth no longer hurts"; positive depth gains
require choices that foreclose (irreversibility), which is a domain property,
not a model defect.
