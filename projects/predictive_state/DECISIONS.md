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
- Compare equal tokens and equal FLOPs; do not interpret the extra predictor's
  compute as free.
- Train fresh probes on frozen checkpoints for representation claims. The
  jointly trained predictor measures system fit, not representation quality.
- Use explicit position IDs in upper-only recurrence; lower-layer caches stop
  growing after prompt prefill and cannot be used as a sequence-length clock.
- Carry a current-segment key mask into upper-only recurrence so packed prompt
  caches cannot leak previous documents.
- Test exact-state scoring before predicted-state planning.
- Label terminal-state geometry as oracle and verified candidate outcomes as
  candidate-privileged.
- Freeze a learned distance during policy optimization and retain the exact
  terminal verifier reward.
