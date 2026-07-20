# Cycle: executable multiscale edit generation

Status: implementation validated; matched training round proposed

## Decision

Can a target-free token prior plus a goal-distance advantage value turn the
validated token/sentence edit dynamics into actual iGSM solution generation,
and does sentence or macro hierarchy improve receding-horizon search over an
information-matched flat token controller?

## Validity boundary

The planner sees the prompt and a masked buffer with official sentence and
token slot structure. It does not see the clean tokens, clean latent, answer,
or target distance. This is therefore **structure-known generation**, not
free-length language generation. Clean sequences enter training only as EMA
dynamics targets and labels for normalized goal distance and exact one-step
token-edit advantage. Counterfactual proposal actions are sampled from the
observable prompt plus current buffer; their clean-relative advantages are
terminal-privileged labels. Evaluation targets never enter the planner API.

## Smallest faithful comparison

Train token-only, sentence-only, token-plus-sentence primitive, and
token-plus-sentence macro models at 10.2M-scale width. Within each job train
three matched prior modes: no learned prior, detached prior, and a prior whose
loss can shape representations. Macro models apply the same mode at both base
and macro levels. Use one epoch as a mechanism screen, fixed learning rate
`3e-4`, EMA, zero dropout, LDAD weight 1 where a sentence representation
exists, no VICReg, and 32 exact counterfactual actions per state.

The first round's full-generation evaluation is only a one-example horizon-1
process smoke. Admit the expensive ID/OOD depth `{1,2,4,8,16}` sweep only for
checkpoints that pass prior accuracy, action-value sign, finite optimization,
non-collapse, and literal-generation process checks. This avoids spending the
16 GPU-hour round allocation on deep search through invalid policies.

## Direction-changing outcomes

- Continue with a learned prior if it improves candidate recall and token
  accuracy over uniform observable-token proposals without damaging latent
  action sensitivity.
- Prefer detached prior training if attached training lowers effective rank or
  transition causality; prefer attached only for a replicated planning gain.
- Continue hierarchy only if a hierarchical checkpoint beats its
  information-matched flat controller at equal candidate width and depth, and
  gains increase rather than disappear with planning depth.
- Stop generation scaling if answer-sentence accuracy remains at chance even
  after prior/content accuracy and action-value sign accuracy are healthy.

No unread steering note was present. The user explicitly requested executable
MPC generation, ID/OOD evaluation, depth ablation, and base/macro prior
ablations. The controller's protected boundary and 16 GPU-hour round
allocation remain in force.

## Nested subgoal CEM and option-decoder extension

The user corrected an important design distinction: a macro decoder is not
required when high-level CEM proposes a sentence latent subgoal and a
goal-conditioned lower planner searches for primitive actions that reach it.
That lower search is the implicit inverse. The implementation now compares
five matched inference paths: primitive beam with macro reranking;
decoder-free subgoal CEM; open-loop macro decoding; closed-loop option
decoding; and decoder proposals followed by lower-level refinement.

The decoder receives the current token state, prompt, macro code, and option
step, and predicts position then content. It is teacher-forced over exact
four-action windows. Detached weight 1 and attached weights 0.1/1 separate
decoder capacity from representation shaping and loss-scale damage. Fifty-five
focused tests plus tiny train/checkpoint evaluations pass. Wave 81 submitted
the detached cell on Lise; its Grünau placement race left the round partial.
Recovery wave 81r submitted the two attached cells on Lise and Grete. Alex was
excluded because its vault is already above the soft inode quota.
