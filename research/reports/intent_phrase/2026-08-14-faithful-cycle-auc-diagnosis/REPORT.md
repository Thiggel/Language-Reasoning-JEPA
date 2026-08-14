# Why LDAD cycle-consistency finds feasibility on stylized iGSM but not on faithful iGSM

Date: 2026-08-14. Every number in this report is a **candidate-privileged,
oracle-labeled diagnostic** (feasibility labels and trajectories come from the
symbolic environment); nothing here is a planning result.

## The question

The cycle-consistency score works like this: take the current latent state,
apply the predictor to a candidate action's embedding to *imagine* the state
change (displacement), then ask the LDAD decoder to read the candidate's own
phrase back out of that imagined displacement (shared path
`src/textjepa/planning/ldad_decode.py`). On STYLIZED iGSM this score
separates feasible from infeasible actions (AUC ~.94, 2026-08-10). On
FAITHFUL iGSM (`hard-ldad-lr3e4-s0-v1`) it is chance — AUC .475 on the
training distribution, .503 on the ID band
(`scripts/probe_faithful_cycle_auc.py`, 24 problems each) — and
`ldad_cycle` planning equals `full_catalogue`. Why?

## 1. Probe-validity control: the probe is fine

New twin probe `scripts/probe_stylized_cycle_auc.py` — the SAME
`delta_logits` + `phrase_log_probs` code path, same oracle random-feasible
walk, same AUC code — on the stylized LDAD checkpoint
(`runs/autonomy/intent_phrase/2026-08-08-intent-stabilizer-sweep-v1/stab-ldad-ema-s0-v1/model/best.pt`):

**pooled AUC .846, per-state mean .865** (24 val problems, 1134 pairs).

Strong separation through the exact code that scores .475 on faithful. The
faithful chance result is real, not a probe bug. (The .846 vs the .94
headline is a protocol difference — random walk over all states vs the
original measurement — the control question is only "signal vs chance".)

## 2. Where the cycle breaks: the PREDICTOR, not the decoder or the phrases

Two further diagnostics on the same checkpoints (oracle trajectories):

| decode of the observed action's phrase from ... | stylized | faithful |
|---|---|---|
| the REAL encoder displacement `s_{t+1} - s_t` | top-1 **1.00** (83/83) | top-1 **1.00** (62/62) |
| the predictor's IMAGINED displacement `P(s_t,u) - s_t` | top-1 **.95** | top-1 **.145** |
| cosine(imagined, real displacement) | **.74** (norm ratio 1.09) | **.11** (norm ratio 1.79, rel. err 2.0) |

And `scripts/probe_faithful_cycle_identifiability.py` (V x V own-phrase rank
at every state): on faithful the imagined displacement identifies the
candidate's own phrase at near-random rank for feasible AND infeasible
candidates (top-1 .10 vs .19, MRR .29 vs .38, V ~= 8-11).

Plain English: on both tracks the LDAD decoder itself is perfect — given the
real state change it names the action every single time. But on faithful the
predictor's imagined one-step state change is essentially noise with respect
to the real one (cosine .11), so the decoder can read nothing out of it, for
feasible and infeasible candidates alike. The cycle score degenerates to
noise → AUC .475. On stylized the predictor's imagined displacement lies on
the real-displacement manifold (cosine .74), the decoder reads the action out
of it (.95), and feasibility emerges because infeasible actions' imagined
displacements fall off that manifold.

## 3. Training-data diff (code audit) — what it is NOT

(a) **Counterfactual/infeasible actions in training.** Both checkpoints
trained with `geo_rank_candidate_interface: feasible_menu` (K=2 feasible
alternatives). The FAITHFUL track additionally injects 2 INFEASIBLE
counterfactuals per ranking anchor (`invalid_counterfactual_k: 2`, consumed
at `src/textjepa/data/faithful.py:410-419`, outcomes rendered with
`step_or_invalid` no-ops). The stylized config carries the same key but
`IGSMDataset` never consumes it (only `geo_rank_invalid_k` under
`full_catalogue`, both off; kwargs list
`src/textjepa/utils/checkpoint.py:173-202`). So faithful saw MORE
infeasible-action exposure than stylized and still scores chance — missing
infeasible training data is NOT the mechanism.

(b) **LDAD sees only observed actions in both tracks** — decoder trained on
encoder displacements of observed transitions
(`src/textjepa/models/discourse_jepa.py:633-637`) against the observed
action phrase (`src/textjepa/objectives/delta_action.py:61-77`). Identical
wiring. The stylized feasibility signal was always emergent, and the table
above shows the emergence lives in the predictor, which the LDAD loss never
touches at the point the cycle uses it.

(c) **Phrase structure differs but is not the primary break.** Stylized:
`derive <v> from <a> <op> <b> .` (parents named,
`src/textjepa/data/igsm/render.py:49-57`); faithful: `Define <owner>'s
<item> .` (target only, `src/textjepa/data/faithful.py:188-189`). This
makes the faithful cycle *weaker in principle* (the phrase carries no
precondition content), but the measured failure is upstream: even the
observed action's phrase — 100% decodable from the real displacement — is
unrecoverable from the imagined one.

Why is the faithful predictor so much worse one-step? Plausible
contributors, unresolved here: harder dynamics at hard caps (op<=21) with
the same latent budget; faithful step sentences contain content that is
unpredictable in principle from the state (the official iGSM renderer draws
the temporary variable letter from a process-global RNG at render time,
`src/textjepa/data/faithful.py:191-195` / `to_sol` — the next state encodes
a random letter the predictor can only average over); and the LR screen is
still mid-flight. The matched-caps retrains will not fix this by themselves
— cos .11 is a dynamics-quality problem, not a caps problem.

## Candidate fixes, ranked by cost

1. **Predictor-cycle LDAD term (cheapest decisive fix; one loss, retrain).**
   Today the decoder trains only on encoder displacements while eval reads
   predictor displacements — the cycle's exact input is never trained.
   Add a term decoding the OBSERVED action's phrase from
   `predictor(s_t, u_t) - s_t` (gradient flowing into the predictor). This
   directly forces imagined displacements onto the decodable manifold —
   the property stylized has for free (cos .74) and faithful lacks
   (cos .11). Self-supervised, observed actions only, no symbolic labels.
   Diagnostic to watch during training: predictor-vs-encoder displacement
   cosine, and own-action decode top-1 from the imagined displacement.
2. **+ counterfactual LDAD contrast (medium; pairs naturally with 1).**
   Faithful batches already carry infeasible counterfactuals
   (`invalid_counterfactual_k=2`, `faithful.py:410-419`). Rank the observed
   continuation's decodability against the counterfactuals' through the
   same predictor-cycle path (observed decodes its phrase, the infeasible
   alternative decodes the "invalid definition" outcome / scores worse).
   This makes the feasibility signal trained-in rather than emergent —
   matching the project's energy-head philosophy (rank counterfactual
   continuations against the observed one) and a cleaner reviewer story.
3. **Compositional faithful action phrases (data-only but least certain).**
   Extend `FaithfulEnv.action_text` to name the dependencies stated in the
   prompt ("Define X using A and B ." — prompt-derivable, no oracle). This
   transplants the stylized phrase mechanism and would sharpen the cycle
   once the predictor is fixed, but by itself it cannot help while the
   imagined displacement is unreadable (the break is upstream).

## Files

- Control probe: `scripts/probe_stylized_cycle_auc.py` (stylized AUC .846).
- Identifiability probe: `scripts/probe_faithful_cycle_identifiability.py`.
- Faithful chance result: `scripts/probe_faithful_cycle_auc.py`
  (.475 train-dist / .503 ID band, CAMPAIGN_LOG 2026-08-14 evening).
- Encoder-vs-predictor decode and cosine numbers: inline diagnostics run
  2026-08-14 (this report, section 2); trivially re-runnable from the two
  probes' building blocks.
