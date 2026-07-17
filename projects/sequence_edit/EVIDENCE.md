# Evidence

- The non-symbolic faithful token-edit data and model path passes local end-to-end smoke tests.
- The first full-scale round is invalid because it failed before training.
- The corrected configuration has not yet produced scientific evidence.
- The token-aligned curriculum EMA control and VICReg weights 0.02, 0.1, 0.5,
  and 1.0 all beat persistence but remain below the causal shuffled-action
  gate. Only 0.02 preserves effective rank within 10% of the control.
