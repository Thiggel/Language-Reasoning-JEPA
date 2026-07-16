# Wave 00 - task, baselines, and evaluation protocol

## Experiments

- Random feasible-action policy.
- Token decoder-only LMs at approximately 2M, 9M, and 27M parameters.
- Sentence LM and sentence LM with next-latent prediction.
- Corrected intent-only token and sentence policies, each scoring the same
  outcome-free feasible intent phrases as JEPA.
- Strict optimal-step and +2-action budgets; validation/test generator split.

Raw logs: `lm_*`, `sent_lm*`, `sentlm_*`, and protocol audit logs in
`../logs/`. Generated comparison: `runs/matched_baselines.md`.

## Principal results

| Model | Strict | +2 actions | Status |
|---|---:|---:|---|
| Random feasible | 0.055 | 0.405 | reference |
| Token intent LM, 3 seeds | 0.827 +/- 0.003 | 0.978 +/- 0.003 | main baseline |
| Sentence intent LM, 3 seeds | 0.690 +/- 0.053 | 0.903 +/- 0.039 | main baseline |
| Sentence+latent, likelihood | 0.817 +/- 0.034 | 0.953 +/- 0.010 | main baseline |
| Sentence+latent, latent distance | 0.840 +/- 0.017 | 0.965 +/- 0.013 | main baseline |

Historical LM results that scored candidate sentences containing computed
outcomes are privileged diagnostics and are excluded from the main comparison.

## Conclusion

The main task is feasible-action ranking, not unrestricted generation. The
matched token and sentence+latent baselines are stronger than originally
reported, so the JEPA recipe is not yet the best overall model.
