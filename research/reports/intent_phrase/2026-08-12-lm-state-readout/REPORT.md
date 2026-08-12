# State-readout probes on LM baselines: the feasibility signal is not JEPA-specific

_2026-08-12. Owner-ordered extension of the 2026-08-11 state-readout controls
to LM baselines. Script: `scripts/probe_state_feasibility_lm.py` (fb08f8e);
raw JSONs in this directory (f4174db). **All labels oracle-derived —
candidate-privileged diagnostic, not a planning result.** Same probes, same
2000/800 problems, same 81,302/33,122 probe rows as the JEPA run; majority
baselines .781 (resolvedness) / .613 (feasibility)._

## Setup

From a frozen state at the same causal boundary (just before the next intent,
no future text) plus a candidate-action embedding, probe (a) resolvedness and
(b) feasibility. Sources: JEPA (stab-ldad-ema-s0, pooled s_t, 256-d), token
LM (runs/lm_intent, last-token causal state, 288-d), sentence LM
(runs/sentlm_intent, next-chunk context latent, 256-d), sentence LM + latent
MSE (runs/sentlm_latent_intent). Candidate embedding: each model's OWN action
representation (`u_own`), plus a matched control using the frozen 16-d JEPA
`u(c)` for all sources (`u_jepa_ref`).

## Results (val AUC, MLP-128; shuffled-state controls flat at .74/.72)

Feasibility: token LM **.971** > sentence LM .961 > sentence LM+latent .947 >
JEPA .935 (`u_own`). With the matched 16-d candidate (`u_jepa_ref`): token LM
.961, sentence LM .934, sentence LM+latent .925 vs JEPA .935 — sentence LMs
land AT the JEPA value, token LM stays above (its 288-d state is an
uncontrolled residual confound). Resolvedness: .968–.982 everywhere,
essentially tied.

Per-depth feasibility (u_own): JEPA decays .965 (leaves) -> ~.72 (depth 4–5);
token LM stays >= .81, sentence LMs >= .83. The one-hot variant (parent names
stripped) collapses feasibility to .76–.78 for ALL sources while resolvedness
barely moves — the conjunctive parent-lookup finding replicates across every
representation family. Latent-MSE auxiliary slightly REDUCES sentence-LM
readability (.961 -> .947; suggestive, single seed).

## Consequence for the paper (important)

Any claim of the form "JEPA states carry prerequisite structure that LM
states lack" is REFUTED — LM baselines encode resolvedness and feasibility at
the same causal boundary as well or better. The 2026-08-11 corrected claim
survives unchanged ("the information is present; imitation-trained readouts
stay near chance; cycle-consistency recovers it through the trained
dynamics"), but the paper's representation story must be about *use*, not
*presence*: every representation family contains the feasibility signal; the
difference that matters is which architecture+objective can convert it into
closed-loop planning (the LDAD cycle route exists only where an
action-conditioned predictor and displacement decoder exist — LMs have no
analogous mechanism, which is measured by the planning tables, not by these
probes). The probes also sharpen an honest JEPA weakness: deep-prerequisite
feasibility decays in the pooled state while LM states hold it — worth
stating rather than hiding.

## Limitations

Single seed per source (JEPA s0, LMs s1); training-campaign confound (LMs:
100k problems x 20 epochs; JEPA: 30k x 10 with ranking anchors) — NOT the
contract's matched-width five-seed protocol; token-LM state width 288 vs 256
uncontrolled; stylized iGSM subsample; oracle labels throughout.
