# Why a codebook of seen actions cannot work here — however large

_2026-08-12. Owner question: "the model should just internally save the exact
latents of the actions it sees — if the codebook is large enough why wouldn't
this work?" Thorough diagnosis on stab-ldad-ema-s0 (frozen), stylized iGSM.
All measurements CPU, scripts inline in session; numbers below are the
complete evidence chain._

## 1. The phrase space does not saturate — it is combinatorial

Unique intent phrases across training problems: 716 after 100 problems,
4,885 after 1,000, 18,571 after 4,000 — still growing linearly (~4.6 new
phrases per problem). Split by type over 300 held-out problems:

| phrase type | seen verbatim in 4k training problems |
|---|---|
| leaf lookups ("look up the number of X") | **1378/1378 = 100%** |
| compute phrases ("derive X from A plus B") | **0/1367 = 0%** |

Leaf phrases are a finite inventory (adjective x noun) and saturate.
Compute phrases name the target AND both parents — a 3-way cross-product
over the name space — and a held-out problem's compute phrases have
essentially probability zero of having occurred verbatim. Consequently **0
of 300 val problems** have their necessary actions covered by any memory of
seen phrases. A planner that can only replay seen actions can never solve a
new problem, at any codebook size: the actions it needs did not exist
before the problem was posed.

This also retroactively explains the generator screen: the sampled phrases
were all leaf lookups — the generator learned exactly the memorizable part
of the action distribution and recombines names for the rest.

## 2. The twist: the LATENTS are nearly reusable — the LANGUAGE is not

Nearest-neighbor distance of a held-out problem's compute-action embeddings
to a 6k-phrase training-embedding memory: **median 0.77**. Separation
between different actions of the same problem: **median 10.69**. So the
16-d action embedding of an unseen action is almost exactly reproduced by
some stored embedding — the owner's intuition is correct at the latent
level. The codebook fails anyway because the environment does not execute
vectors, it executes sentences: the retrieved neighbor's *own phrase* names
a different problem's variables (parse rate 0 for compute actions), and
decoding the vector through the LDAD decoder reproduces that wrong-problem
phrase. Storing (phrase, latent) pairs — the cleanest "saved action menu" —
fails identically, because 0% of needed compute phrases are in the store.

Corollary about the representation: the action embedding is a coarse
role/operation code (which is why one vector serves many name-combinations,
and why cycle-consistency — which only *scores a given phrase* — works at
AUC .94), not a container for the surface names. The names live in the
state (the frozen-state decoder reads them at 96% token accuracy); no
current proposer head extracts them into new phrases.

## 3. Conclusion

"Save what you saw" is impossible in this environment for a fundamental
reason, not an engineering one: the intent-action space is compositional,
and evaluation problems are new compositions by construction. Any
catalogue-free proposer must *construct* phrases from the current prompt
(names) plus learned structure (operations) — a token-level generation
ability — or the planner must enumerate the catalogue from the prompt text,
which is the ldad_cycle protocol we report. For the paper this is a
positive framing: the environment's action space has the combinatorial
character of language, so retrieval-based action memories are ruled out in
principle, and the menu-free result via cycle-consistency is the right
non-trivial claim.

Limitations: one checkpoint (s0); 4k-problem phrase census (growth is
linear, so larger censuses only strengthen the claim); embedding distances
on a 6k-entry memory.
