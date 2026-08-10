# Decisions

- Freeze embeddings and the lower target stack in Stage 1; detach action
  embeddings so the auxiliary loss cannot rewrite token identity.
- Capture residuals directly from decoder-block outputs, avoiding ambiguity in
  whether a library's final `hidden_states` entry includes the terminal norm.
- Use parameter-free RMS normalization for cosine prediction and retain a
  separate raw activation-scale loss for injectable states.
- Compare equal tokens and equal FLOPs; do not interpret the extra predictor's
  compute as free.
- Train fresh probes on frozen checkpoints for representation claims. The
  jointly trained predictor measures system fit, not representation quality.
- Use explicit position IDs in upper-only recurrence; lower-layer caches stop
  growing after prompt prefill and cannot be used as a sequence-length clock.
- Test exact-state scoring before predicted-state planning.
- Label terminal-state geometry as oracle and verified candidate outcomes as
  candidate-privileged.
- Freeze a learned distance during policy optimization and retain the exact
  terminal verifier reward.
