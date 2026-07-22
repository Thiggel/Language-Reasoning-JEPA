# Token-level prerequisite support for learned intent catalogues

## Decision

Determine whether token-level causal matching between a candidate intent and
the executed intent history reduces invalid actions on length-nine problems
beyond phrase-pooled support and a history-masked, equal-capacity control.

## Validity audit

- All four controller jobs completed at commit
  `2bdd5a01b5f987889147d39b8af7808e8b1642f8` with exit code zero and every
  declared artifact present.
- The three trained cells used the same seed, source checkpoint, 40,000
  examples, five epochs, frozen JEPA encoders/dynamics, catalogue prior, and
  zero dropout. They varied only support architecture history access and the
  predeclared learning-rate cross-check.
- Evaluation used the same 120 episodes per length/budget, stable seed 7321,
  shuffled learned catalogues, top four proposals, no future-action oracle,
  and prior-only selection. Consequently this round tests proposal support,
  not JEPA reranking or latent simulation quality.
- No NaNs, exceptions, missing checkpoints, or non-benign warnings were
  observed. The repeated Transformer nested-tensor warning is unrelated to
  numerical validity.
- This is a one-training-seed pilot. Support-weight selection from the same
  evaluation episodes is exploratory and must not be treated as a sealed test.

## Observations

- At length nine, the best observed aligned token cell reaches `.175` strict
  and `.717` slack-two success, versus `.083` and `.358` for the best observed
  phrase-pooled cells. The history-masked token control reaches at most `.025`
  on either budget.
- For the corresponding strongest cells, length-nine invalid-action rates
  fall from approximately `.48-.51` for phrase pooling to `.16-.19` for token
  history; the masked control remains at `.95-.99`.
- The aligned token support audit is almost invariant to true, one-step
  predicted, or open-loop predicted states. At learning rate `1e-3`, length-nine
  support accuracy is `.992` and pair accuracy is `.996` in all three modes.
  Phrase pooling reaches about `.947-.958` accuracy, while the masked token
  control falls to `.724` open-loop accuracy.
- Learning rate `1e-3` is behaviorally better than `3e-3` for the aligned token
  head despite both attaining high support classification accuracy. This is a
  calibration/ranking warning: aggregate binary accuracy alone does not select
  the best closed-loop controller.
- The best token prior remains below the information-matched first-feasible
  reference at length-nine slack two (`.717` versus `.792`), although it exceeds
  that reference in strict success (`.175` versus `.142`).

## Inference

Lexically preserving the executed intent history resolves most of the learned
catalogue's executability problem. Phrase pooling discards prerequisite identity
information that the support head needs, and the history-masked control shows
that the gain cannot be explained by the larger token-attention head alone.

This does not yet show a JEPA planning gain: the deployed score in this round is
the behavior-cloned catalogue prior plus learned feasibility support, and
`proposal_rerank_weight=0`. The token head should therefore enter the proposal
interface, while the next decision must ask whether JEPA reranking adds useful
ordering after invalid actions are no longer the dominant failure.

## Next decision

Hold the selected `1e-3` aligned token-support checkpoint fixed and evaluate a
small, matched inference factorial over rerank weight and simulation depth,
including prior-only, one-step JEPA, and deeper causal rollout. Use the same
catalogue, top-M, beam width, episodes, and support weight. Direction changes
only if JEPA reranking improves success without increasing invalid actions; if
it does not, diagnose value error versus recursive dynamics drift before any
new training or scaling.

