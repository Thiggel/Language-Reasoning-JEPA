# Evidence

- Token-aligned operation, current-buffer pointer, and local content all pass
  direct component-shuffle falsifiers.
- Total exposure, rather than unique clean-problem count up to 18k, drives the
  current data improvement; the 18k effect replicates over three seeds.
- D512 improves d256 under matched batch and optimizer-step comparisons.
- Exact counterfactual breadth saturates at K=1; K>1 is rejected, and K=1's
  recursive-error cost remains unresolved.

- The non-symbolic faithful token-edit data and model path passes local end-to-end smoke tests.
- The first full-scale round is invalid because it failed before training.
- The corrected configuration has not yet produced scientific evidence.
- The token-aligned curriculum EMA control and VICReg weights 0.02, 0.1, 0.5,
  and 1.0 all beat persistence but remain below the causal shuffled-action
  gate. Only 0.02 preserves effective rank within 10% of the control.
- Faithful text LDAD weights 10 and 20 substantially reduce one-step and
  recursive error and retain effective rank, but all five LDAD cells remain
  action-blind at shuffled/matched 1.006--1.011.
- The mask-only checkpoint's provisional 1.163 action-sensitivity ratio falls
  to 1.003 on mixed edits. This 64-example cross-evaluation is provisional but
  demonstrates that native-regime evaluations cannot select a transferable
  corruption recipe.
- Non-curriculum token-edit datasets repeated corruptions across epochs despite
  the configured fresh-data flag. The option is now explicit and tested; old
  runs remain labelled as repeated-corruption baselines.
