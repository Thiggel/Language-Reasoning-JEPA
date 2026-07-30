# Decisions

- Preserve all historical hard-text logs, checkpoints, plans, and run-family
  names in place as legacy.
- Use a frozen causal reasoning LM as the wide encoder and executable proposal
  distribution; train separate token and sentence planning projections.
- Admit one falsifiable mechanism at a time. First test token JEPA with a true
  future goal and exact endpoint control; next test a token worker against a
  true sentence waypoint.
- Keep oracle terminal, symbolic, candidate-privileged, and cross-project
  information explicitly labelled.
- Do not add latent macro-actions, high-level CEM, or value distillation until
  both decisive worker gates pass.
- Do not include a remaining-budget input. `K0`, `K1`, `KV`, and `n_exec`
  retain distinct planning, teacher-computation, and execution meanings.
- Require information-matched controls, shortcut/cache diagnostics,
  compute-matched depth comparisons, exact grounding, and optimizer-curse
  curves.
