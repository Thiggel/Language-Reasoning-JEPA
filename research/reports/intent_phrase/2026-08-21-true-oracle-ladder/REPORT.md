# True-oracle ladder: search exonerated, the learned ruler carries the entire depth collapse

Date: 2026-08-21 · faithful iGSM-med, 200 val episodes ID, checkpoint
`_ckpt_snapshots/ecf16-s0-best-2026-08-20.pt` · run dir
`runs/autonomy/intent_phrase/2026-08-20-true-oracle-upper-bound/` (HANDOFF.md
there has the ladder spec; `summarize.py` reproduces every table) · scorer
implemented in commit 90cc2ac (`--scorer symbolic_oracle` in
`src/textjepa/planning/flat_search.py`, 8 contract tests).

**Evidence label: the symbolic-oracle rows are a CANDIDATE-PRIVILEGED ORACLE
DIAGNOSTIC, evaluation only.** The oracle executes each candidate sequence in a
cloned env and ranks by remaining necessary steps (solved = −1, tie-broken
toward fewer no-progress steps). It never touches training and is never a model
component.

## What the experiment is

The same beam/MPC search machinery is run with four different "rulers", from
fully symbolic to fully learned, so the depth-collapse blame decomposes
cleanly:

1. **symbolic oracle × true executed endpoints** — perfect ruler, perfect
   endpoints: measures the *search procedure itself*.
2. **latent oracle-distance × true executed** — our encoder's L2-to-solved-state
   ruler on real states: measures the *representation*.
3. **latent oracle-distance × imagined** — adds the predictor's imagination
   error.
4. **learned energy × imagined** — what we actually ship.

## Headline result

**Search is completely exonerated.** Symbolic oracle × true endpoints solves
1.000 of episodes in exactly the minimum number of steps at every depth
(1/2/4/8), at every cap including strict 1.0, with zero invalid and zero
distractor executions, on `full_catalogue`. Beam width is irrelevant
(max_expand 8 = 64 = 256 → 1.000). Everything lost at depth is lost by the
learned rulers.

`full_catalogue`, success at cap 1.0 (cap 1.25 in parentheses):

| ruler × endpoint | d1 | d2 | d4 | d8 |
|---|---|---|---|---|
| symbolic × true (upper bound) | 1.000 | 1.000 | 1.000 | 1.000 |
| latent-dist × true (representation) | .320 (.650) | – | .033 (.050) | .040 (.080) |
| latent-dist × imagined (+ predictor) | .175 (.330) | .150 (.315) | .035 (.155) | .010 (.085) |
| learned energy × imagined (shipped) | – | .585 (.720) | – | .120 (.220) |

Two decomposition facts worth stating in the paper:

- The latent goal-distance ruler is *worse* on true executed states than the
  learned energy is on imagined ones — "oracle_distance" was never an upper
  bound, it measures representation geometry (this retires that column for
  good).
- At the generous cap 4.0 all learned rows recover to .77–.96, which is why the
  collapse was invisible under the old protocol.

## prior_propose: the ceiling is the proposer, and only the proposer

Symbolic oracle × true endpoints with the token-head proposer: d1 **.945**, d2
**.920**, d4 .830–.840, d8 .830. Failure accounting of every unsolved episode:
d1 11/11, d2 16/16, d4 ~32–34, d8 34 — **all** proposal exhaustion (the needed
action was never generated at some state); zero losses to budget, invalid
execution, or wrong choices among proposals. Per-step proposal recall is
.98–.99, but one proposal-dead state kills an episode. Proposal *coverage* is
therefore the single number that separates any proposer-based interface from
its ceiling.

Note the mild depth decline (.945 → .830) even with a perfect ruler: deeper
search executes longer committed prefixes between re-plans, so a single
proposal-dead state costs more. This is a proposer effect, not a search effect.

## Design note

"Symbolic oracle × imagined endpoints" is not definable — an imagined latent
has no symbolic state. The predictor's contribution is isolated instead by the
lat-true → lat-imag contrast with the ruler held fixed. The lat-true rows at
d4/d8 are partial (n=25–60; they require executing every candidate, ~18h/run)
and are labeled as such.

## Consequences in force

1. The depth fix must target the learned ruler (energy head / representation on
   off-reference states) — see the companion report
   `2026-08-21-imagined-energy/`.
2. The paper gets a clean three-way attribution: search 0% / ruler ~all /
   proposer = coverage-only, each measured and labeled.
3. `oracle_distance` is renamed conceptually to "latent goal-distance (self)"
   and never again described as an upper bound.
