# Open-ended proposer screen: can we plan with no action catalogue at all?

_2026-08-11. Screen of three catalogue-free proposal interfaces against the
established menu-free reference (`ldad_cycle`, which enumerates the problem's
variable catalogue and filters it by LDAD cycle-consistency). All on the LDAD
headline recipe; iGSM val episodes; invalid=noop; no feasibility oracle, no
symbolic menus. Run dirs:
`runs/autonomy/intent_phrase/2026-08-11-intent-openended-screen-v1/`._

## What was screened

1. **generator_cycle** (`gen-ldad-s0-v1`): LDAD recipe retrained with a small
   state-conditioned autoregressive head that writes intent phrases token by
   token (16 nucleus samples per state; PAD-stop supervised after the
   adversarial-review fix in 318d6c2), candidates vetted by cycle-consistency.
2. **cem_cycle** (`eval-cem-s0-v1`): eval-time CEM directly optimizing 16-dim
   action embeddings (pop 64, 8 elites, 3 iters; diagonal-Gaussian prior
   fitted on training-problem embeddings; off-manifold penalty), elites
   decoded to phrases via the LDAD displacement decoder (with the '.'
   terminator truncation fix, snapshot 734bc58). No retraining.
3. **codebook_cycle** (`eval-codebook-k{64,128}-s0-v1`): eval-time k-means
   codebook over training action embeddings, codes decoded to phrases.

## Results — all three fail, all for the same reason

| interface | parse rate | usable proposals/state | no-proposal episodes | strict |
|---|---|---|---|---|
| generator_cycle | .132 | 1.0 of 16 | 98.7% | .013 |
| cem_cycle | .032 | 0.2 of 8 elites | 100% | .000 |
| codebook_cycle k64/k128 | ~0 pre-fix | ~0 | ~100% | .000 |
| ldad_cycle top-2 (reference, 2026-08-10) | — (catalogue) | — | — | .027 strict / .29 slack-4 |

(Random policy WITH the oracle menu: .050 strict — every catalogue-free
variant is below it.)

**The shared failure is grounding, not fluency and not search.** Sampled
generator phrases and decoded CEM/codebook phrases are fluent, correctly
formatted intent phrases — but they name variables that do not exist in the
current problem (adjective/noun recombinations across the training
distribution, e.g. "square beads" for a problem with neither word), and the
generator emits only leaf lookups. Lowering sampling temperature shrinks the
set without raising the parseable fraction. Generator training itself was
healthy (CE plateau 0.62/token; backbone metrics unchanged).

## Interpretation

Emitting *this problem's* variable names from the pooled state is the single
bottleneck for every catalogue-free proposer. This is the same
conjunctive state-action binding gap the state-readout controls localized
(2026-08-11-state-readout-controls): the information is present (oracle
probe AUC .935; the frozen-state sentence decoder reads variable names from
true states at 96% token accuracy), but heads trained with generative /
imitation objectives do not extract the problem-specific inventory. Note the
asymmetry with the state decoder: the decoder reads names *recorded in* s_t
(the executed step); a proposer must select the *next* variable, which
requires the feasibility computation on top of the inventory.

## Conclusion for the paper

The menu-free axis stands as: **catalogue enumeration (trivially derivable
from the prompt text) + LDAD cycle-consistency feasibility** — no oracle at
plan time. Fully catalogue-free proposal (generator, CEM-in-embedding-space,
codebook) fails uniformly on grounding; we report this as a
precisely-characterized negative with the shared diagnosis above, not as
three unrelated failures. A prompt-reading proposal module would fix it but
is a token-level component, i.e. a different contribution.

Evidence labels: all numbers are plan-time-oracle-free; proposal_recall uses
oracle labels for measurement only. Caveat: proposal_parse_rate denominators
differ across interfaces (generator = all samples, CEM = elites) — not
comparable raw, both are far below usable either way. Single seed (s0).
