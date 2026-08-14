# Decisions

- Freeze embeddings and the lower target stack in Stage 1; detach action
  embeddings so the auxiliary loss cannot rewrite token identity.
- Capture residuals directly from decoder-block outputs, avoiding ambiguity in
  whether a library's final `hidden_states` entry includes the terminal norm.
- Use parameter-free RMS normalization for cosine prediction and retain a
  separate raw activation-scale loss for injectable states.
- Apply `λ_pred` and `λ_scale` independently; a regression test forbids nesting
  the scale term under the prediction coefficient.
- Reset packed position IDs at document boundaries and pass no 2D padding mask,
  allowing Transformers to construct block-diagonal causal attention. WikiText
  document units are top-level articles, not isolated paragraphs.
- Never unfreeze the stack below the prediction target. A trainable target can
  drift to meet the predictor, which collapses the auxiliary loss without
  encoding anything; the sibling subprojects fixed exactly this failure with a
  frozen anchor. Upper-half LoRA is how representations are allowed to move.
- Size the corpus so a token budget stays under one epoch. WikiText-2 holds
  2.3M tokens, so the 20M-token screen ran on WikiText-103 articles instead;
  otherwise held-out NLL movement is confounded with memorization.
- Every cell in a comparison reads one pre-built token-block file rather than
  rebuilding its own, so train and validation tensors are identical by
  construction and not merely by matching digests after the fact.
- Compare equal tokens and equal FLOPs; do not interpret the extra predictor's
  compute as free.
- Train fresh probes on frozen checkpoints for representation claims. The
  jointly trained predictor measures system fit, not representation quality.
- Use explicit position IDs in upper-only recurrence; lower-layer caches stop
  growing after prompt prefill and cannot be used as a sequence-length clock.
- Carry a current-segment key mask into upper-only recurrence so packed prompt
  caches cannot leak previous documents.
- Weight the next-token term explicitly. It carried a hard-coded weight of
  1.0, so a prediction-weight sweep could not reach auxiliary dominance and
  a negative from such a sweep did not mean what it appeared to.
- Report held-out next-token loss unweighted whatever the training weights,
  so cells that optimize it at different strengths stay comparable.
- Size the corpus so a token budget stays under one epoch, and prefer a
  corpus close to the model's own pretraining distribution: on WikiText-103
  part of the measured cost was domain adaptation rather than the objective.
- Download a fixed corpus subset rather than streaming one. Every cell
  records a dataset digest and a stream has none.
- No symbolic or hardcoded heads in trained components, per the 2026-08-14
  rule. Steps-to-go regression and verified-outcome ranking labels are
  evaluation-only; the energy head ranks counterfactual continuations
  against the observed one.
- Test exact-state scoring before predicted-state planning.
- Label terminal-state geometry as oracle and verified candidate outcomes as
  candidate-privileged.
- Freeze a learned distance during policy optimization and retain the exact
  terminal verifier reward.
