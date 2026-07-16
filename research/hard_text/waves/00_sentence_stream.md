# Wave 00 - sentence-stream formulation

The prompt and reasoning sentences were packed into one causal stream with no
observed intent/outcome boundary. The state encoder and predictor were tested
on stylized and faithful iGSM sentence streams.

The formulation established the data path, causal masking, Gaussian
next-state model, and action-free diagnostics. It did not yield a competitive
planner because no reliable plan-time action proposal existed.

Raw logs: `sentence_vjepa_stylized.log` and
`sentence_vjepa_faithful.log` in `../logs/`.
