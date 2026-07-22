# Replacement-only interface audit and original-scale design

## Decision

Do not scale the current multiscale implementation unchanged.  First remove
absolute-coordinate action text and make location a structural slot selection.
The paper-scale comparison is admitted only after the corrected small models
pass memorization, causal-action, proposal-support, and generation gates.

## Observed implementation problems

- The edit vocabulary adds 1,001 action-only integer strings (`23`--`1024`).
  They enter MDLM/JEPA softmaxes despite not being clean-solution tokens.
- Iterative-refinement noise samples uniformly from that full vocabulary, so
  action-only integer strings can be inserted as synthetic wrong words.
- Observed-action LDAD reconstructs `replace token position N with word .` and
  therefore explicitly rewards a sentence displacement for encoding `N`.
- The replacement prior learns a single expert next-position label even though
  every unresolved correct replacement is a valid denoising action.  This
  penalizes alternative valid orders.
- The token predictor receives the full-dimensional content embedding through
  its edited scaffold in addition to the small action code.  Thus the nominal
  action bottleneck does not bottleneck token identity.
- The active token state is an array of `length × d_model` contextual vectors,
  not a small global state bottleneck.
- Latest multiscale batches contain four trajectories per problem; batch eight
  therefore means only two independent clean problems.  VICReg additionally
  treats correlated tokens/snapshots as its sample population.
- One trajectory slice start is selected for the entire padded batch; shorter
  examples can contribute no valid transitions for late slices.
- Beam depth mechanically edits and re-encodes each literal child buffer.  It
  uses learned one-step scores at each node rather than a recursive JEPA latent
  rollout, so depth is not currently evidence for accurate multistep world
  modeling.
- Sentence boundaries and exact response shape are supplied from the hidden
  target.  MDLM receives total response length but not the sentence partition,
  so existing cross-architecture comparisons are not information matched.

## Corrected action interface

- A primitive candidate is externally indexed as `(slot, replacement_token)`.
  The learned action content is only a bottlenecked token code.
- Token-only: scatter the action code at the selected token slot or mark that
  slot structurally; do not embed or decode an integer position.
- Sentence-only: gather the selected token's contextual representation, combine
  it with the bottlenecked content code, and inject the result into the affected
  sentence slot.  The pointer is used only for gathering and routing.
- Token+sentence: route through the selected token slot, then attention-pool the
  predicted lower state.  Do not redundantly provide an absolute coordinate to
  the sentence level.
- Do not train a single-label position policy in the core comparison.  Enumerate
  unresolved slots or use a target-independent schedule; predict token content
  per slot.  Position pruning is a later efficiency ablation.
- LDAD reconstructs the bottlenecked content action (and a constant replacement
  operation only if retained), never textual coordinate tokens.

## Independent-example data view

Use one trajectory per problem and one randomly selected transition/window per
problem per optimizer step.  Every effective-batch member is a different fresh
iGSM problem.  Counterfactual candidates remain explicitly within-anchor data
and are never counted as independent batch items.

## Original-scale comparison design

Primary regime matches the iGSM-medium pretraining reference: fresh procedural
problems, batch 512, 100,000 optimizer steps, peak LR 0.002, AdamW betas
`(0.9, 0.98)`, weight decay 0.05, 1,000-step warmup, cosine decay to 1% of peak,
context 768, and train support through 15 operations.  Four methods receive the
same prompt, response budget, visible structural separators, update count, and
independent examples:

1. absorbing-mask MDLM with faithful SUBS and categorical reverse sampling;
2. token-state JEPA;
3. attention-pooled sentence-state JEPA;
4. token-to-sentence JEPA.

Target approximately GPT-2-small total trainable capacity for every method.
Keep twelve total learned Transformer blocks per forward path and adjust widths
or small feed-forward dimensions to match trainable parameters within 1%; do
not count frozen EMA copies.  Keep action/state projection bottlenecks fixed and
report them separately.

An exactly matched LR anchor is required by the requested comparison, but it
cannot establish method fairness by itself because likelihood and latent losses
have different scales.  Before committing 100,000 steps, each method gets a
bounded LR validity screen at 1,000--3,000 steps.  A method-appropriate LR
companion is retained if the shared `0.002` anchor is unstable or collapsed.

## Admission gates

- One-example and one-minibatch memorization succeed.
- No action-only coordinate token occurs in a state, corruption, content
  proposal, or reconstruction target.
- Shuffling token content or slot routing materially worsens transition error.
- Content top-M recall beats the marginal-token baseline at fixed mask rates.
- Generation is evaluated with the same visible boundary information for every
  method.
- Paper-scale execution requires an explicit approved run plan and budget.

