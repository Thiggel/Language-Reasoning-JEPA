# PlanBench + ALFWorld unlock: data gates pass, learning gate fails — admission decision needed

_2026-08-12. Adapter completion + admission gates for the two remaining
contract domains. Corpora: PlanBench Blocksworld 3000/60/80 (+48-episode
length-OOD), renaming-invariant split identity, provenance MANIFEST;
ALFWorld `data/intent_phrase/alfworld/paper_v1/` 840/80/80 episodes,
12204/1111/1244 transitions, 48816/4444/4976 counterfactuals (half
deliberately infeasible), all 6 task families in every split, official
train/valid_seen/valid_unseen, zero gamefile overlap. Full details in the
campaign log entry (commit 3e1849b); gate run dirs under
runs/autonomy/intent_phrase/2026-08-12-intent-domains-v1/._

## Infrastructure defects found and fixed on the way (all committed)

- The observed-action schema never recorded teacher-rollout ACTION codes,
  so geo_horizon_rank — the headline recipe's main ranking loss — silently
  evaluated to zero on ALL compiled domains (ProofWriter included). Fixed
  (2c405d9, previous agent); verified live: 48816/48816 ALFWorld
  counterfactual branches carry teacher_rollout_actions.
- build_dataset now raises instead of silently ignoring dropped data flags
  (same defect family as the faithful-adapter shuffle bug).
- Gate tooling hardcoded iGSM's max_chunk_len=96; ALFWorld renders longer,
  crashing the encoder — the ALFWorld gate could not run at all (053d194).
- 47 lingering TextWorld collector processes (post-write cleanup hang)
  starved live shards; ALFWorld corpus frozen at 840 train episodes (40/48
  stripes; contributing shards listed in MANIFEST).

## Gate outcomes

Data-side: BOTH domains pass everything — exact outcome replay 1.0, expert
action in grounded catalogue 1.0, executor reaches goal 1.0, disjoint
splits, horizon Energy live, random 0.0 / oracle 1.0 bounds, dropout/EMA
invariants.

Learning-side tiny-overfit: BOTH domains FAIL on strict success. PlanBench:
0.36M model, 400 epochs, strict 0.0 ON ITS OWN TRAINING SET while
memorising it (sequence_exact .702, token acc .950); invalid-action rate
92% on the full catalogue. ALFWorld: strict 0.0, sequence_exact .534.

Decisive diagnostic (length-2 plans only, so success is not floored by
compounding): goal_dist_corr .995 (near-perfect geometry) yet strict .167
vs random .250 — endpoint-Energy SELECTION is below random at tiny scale
even though representation learning and the action decoder both work. The
shuffle falsifier bites hard on the non-floored metrics (sequence_exact
.702 aligned vs .013 shuffled) and reads correctly on the length-2 set.

## Consequence + owner decision needed

Per PAPER_EXPERIMENTS.md admission rules, neither domain is admitted to
the LR sweep. OPEN QUESTION FOR THE OWNER: is tiny-overfit-on-strict-
success the right bar for domains with 7-15-step plans (where a tiny model
can memorise the text but a single selection error zeroes strict success),
or should admission score a non-floored metric (sequence-exact /
goal_dist_corr / soft success), which both domains clearly pass? Prepared
cells (planbench-ldad-lr3e4-s0-v1, alfworld-ldad-lr3e4-s0-v1, snapshot
3e1849b) stay PREPARED until decided.

Still running detached: ALFWorld shuffled cell, full-catalogue eval (up to
2895 candidates), closed-loop val evals (_gate_alfworld/finish_gate.sh;
gate_summary.json on completion).
