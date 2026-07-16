# Wave 01 - action-free variational latent actions

## Experiments

Target mode (EMA/online stop-gradient/fully online), regularizer
(none/VICReg/SIGReg), and latent-displacement consistency off/on were crossed
for the sentence stream.

Generated tables: `runs/variational_factorial.md` and
`runs/discourse_variational.md`.

## Conclusion

Healthy global variance or effective rank did not imply same-state action
identification. The transition-informed posterior could reconstruct observed
transitions, while the state-conditioned prior failed to cover or select
distinct feasible continuations. This track is paused rather than treated as
the default hard-project recipe.
