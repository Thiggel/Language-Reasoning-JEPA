# Flat-backbone intent JEPA (2026-08-19)

Code: `src/textjepa/models/flat_intent_jepa.py`, `src/textjepa/data/flat_stream.py`,
`src/textjepa/planning/flat_search.py`, `scripts/train_flat_jepa.py`,
`scripts/plan_flat.py`, `scripts/probe_flat_energy_auc.py`, `configs/flat_jepa.yaml`.

## One backbone, one stream

The intent JEPA no longer has a chunk encoder + pooled sentence vectors +
causal state model.  A single causal token transformer (the token-LM
architecture: d768 / 12 layers / 12 heads, 91M) reads the trace exactly as
the token LM is trained on it:

    prompt  intent_1  outcome_1  intent_2  outcome_2  ...  intent_T  outcome_T

and is initialized from the trained token LM checkpoint
(`2026-08-18-lm-med-scale-v1/tok-lm-med-fullsol-big-lr1e3-s0`, free-gen
success .825 on iGSM-med) — `model.init_from_lm`, with `encoder_mode`
full | frozen | lora.  This makes the JEPA backbone-matched to the LM rows.

* **State** `s_t` = hidden state at the last token of `outcome_t`
  (`s_0` = last prompt token).  The **goal / solved state** is the same
  encoder on the completed trajectory (`s_T`), EMA-teacher side, used only in
  Energy labels — as in the old recipe's `geo_energy_target=distance`.
* **Action** `a_t` = hidden state at the last token of `intent_t`, i.e. the
  intent phrase encoded **in context** (it sees the prompt and the history
  through attention).  Candidate intents are encoded the same way with a
  *phrase block*: the problem's action catalogue is appended after the
  stream with an attention mask that lets each phrase see the prefix up to
  the anchor state and its own earlier tokens (one forward pass for all
  candidates, identical to how the planner encodes candidates at a state).
  Actions inside imagined rollouts (depth >= 2) are encoded in the context
  of the real anchor prefix (there is no text for imagined steps).
* **EMA teacher** = EMA copy of the whole encoder (momentum .99 -> .999).
* **Predictor** F(s_t, a_t) -> s_{t+1}: residual MLP on `[LN(s); a]`
  (default, `predictor_kind=mlp`) or the causal history predictor
  (`predictor_kind=causal`, option).  Targets are LN-normalized EMA states.
* **Energy**: horizon-blind `HorizonEnergyHead` E(root, endpoint, s_0).

## Losses (ported 1:1 where they apply; weights in `configs/flat_jepa.yaml`)

| term | what | w |
|---|---|---|
| latent_pred | F(s_t,a_t) vs EMA s_{t+1} (smooth-L1 on LN) | 1 |
| vicreg | variance/covariance on states (+0.1 on actions) | 1 |
| counterfactual_state | F(s_t*, u_cf) vs EMA state after u_cf's *true* outcome (K alternatives at one anchor per trace) | 1 |
| geo_horizon_rank | logistic ranking of E(s_t*, imagined endpoint after H predictor steps, s_0) over all candidates x R=4 random rollouts at one horizon in {1,2,4,8}; labels = LN-L1 distance of the EMA-encoded TRUE rollout endpoint to the encoded solved state | 1 |
| geo_advantage_mse | pair-difference regression of the 1-step energies onto best-of-R label differences | 0.25 |
| observed_action_ldad | decode `intent_t` tokens from `s_{t+1} - s_t` | 1 |
| **energy_cf_feasibility_rank** (new, central) | softplus(E(F(s,u_obs)) - E(F(s,u_cf))) through the same head, (i) at the anchor against the counterfactual pool, (ii) **along every imagined prefix** of the training rollouts: at depth h the "observed" continuation is the rollout's actual next action, the counterfactuals are infeasible intents at that rollout state (`rollout_counterfactual_k`, half premature / half already-resolved), root = real anchor — exactly how oracle-free lookahead scores deeper slots | 4 |
| **intent_prior_lm** (new) | next-token CE on the stream (`lm_loss_on=all_solution`: intent + outcome tokens; `intent`: intents only) | 1 |

The counterfactual pool at the anchor: 2 feasible alternatives + 8
premature-infeasible (unresolved-only hard negatives) + 4 already-resolved
(easy negatives) — the full legality boundary.  Label = which continuation
occurred / what the executor returned; no symbolic state anywhere.

**chunk_pred is gone**: it anchored the state space to frozen chunk
embeddings to prevent encoder/predictor collusion; the token-level CE of the
intent prior anchors the same states to the text itself, and is also the
generative proposer for menu-free planning.

## What differs from the old recipe and why

* Backbone: pretrained token LM instead of a from-scratch chunk+state stack —
  parameter- and data-matched to the LM baselines, and states are the LM's own
  states (the "what do LM states already afford" question is the frozen cell).
* Actions are contextual (cross-sentence attention) instead of context-free
  bottleneck codes.
* Feasibility is learned by the planning Energy itself (new contrast term,
  also along imagined prefixes) instead of by a separate LDAD-cycle gate; the
  old stylized head was anti-feasible (AUC .39/.32) and its depth gains
  needed symbolic future menus.
* Menu-free proposals come from the backbone's own next-token head
  (`prior_propose` / `autonomous`), ranked by the Energy; codebook kept as a
  comparison interface.

## Planning / eval (`scripts/plan_flat.py`)

Interfaces: feasible_menu, full_catalogue, ldad_cycle, codebook_ground,
prior_propose (K=16 samples: 1 greedy + nucleus p=.9, parse, ground to the
env's action text, Energy beam), autonomous (prior_propose + the backbone
writes the outcome sentence; env only grades the goal).  No scored budget;
runaway cap 4x necessary; success + steps-used distribution; depths {1,4,16}
with oracle-free lookahead (deeper slots from the root pool, ranked by the
same Energy; `imagined_invalid_rate` logs the legality of chosen imagined
slots as a diagnostic).  Reference rows: random / first-feasible under the
same interface rules.  Probes (`probe_flat_energy_auc.py`): energy AUC at
depth 0 and along 1-step imagined prefixes, plus LDAD cycle AUC.
