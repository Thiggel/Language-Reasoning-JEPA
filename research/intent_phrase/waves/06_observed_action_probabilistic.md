# Wave 06 - observed-action probabilistic dynamics and faithful LDAD

## Experiments

Deterministic and probabilistic-state factorials crossed target mode
(EMA/online stop-gradient/fully online), regularizer
(none/VICReg/SIGReg), and faithful raw-intent LDAD off/on. Follow-ups tested
residual/direct predictors and pooled/ordered-token action encoders.

Generated tables: `runs/observed_ldad_factorial.md`,
`runs/observed_vjepa_ldad_factorial.md`, `runs/variational_architecture.md`,
`runs/variational_transfer.md`, and `runs/variational_rollout.md`.

## Principal official probabilistic-state results

| Architecture | LDAD | Matched L1 | Shuffle ratio | State rank |
|---|---:|---:|---:|---:|
| EMA+SIGReg residual | off | 0.463 | 1.105 | 119.7 |
| EMA+SIGReg residual | on | 0.067 | 1.462 | 102.9 |
| EMA+SIGReg direct | off | 0.719 | 1.097 | 185.3 |
| EMA+SIGReg direct | on | 0.266 | 1.457 | 163.5 |
| EMA+VICReg direct | on | 0.199 | 1.307 | 197.3 |

## Conclusion

Faithful LDAD improves conditional transition means and action sensitivity,
but does not replace EMA or distributional regularization. This track is now
paused and is not part of the active hierarchy search.
