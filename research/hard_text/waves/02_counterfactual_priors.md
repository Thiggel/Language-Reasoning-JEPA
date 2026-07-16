# Wave 02 - same-state counterfactual outcome priors

## Experiments

- Single-Gaussian versus mixture action priors.
- Full same-state rendered outcome-set supervision on/off.
- Posterior-code reconstruction and 64-sample prior coverage audits.

Generated table: `runs/action_free_transfer.md`.

## Principal result

The strongest online-stop-gradient+SIGReg model had broad true-outcome
separation and approximately 0.598 broad prior-region coverage, but only
0.073 posterior-accuracy coverage. Posterior-code reconstruction improved
broad coverage to approximately 0.675 while reducing precise coverage to
0.029.

## Conclusion

More prior capacity and posterior reconstruction did not solve precise
action-free control. The remaining jobs were stopped when multiscale observed
token actions became the active direction.
