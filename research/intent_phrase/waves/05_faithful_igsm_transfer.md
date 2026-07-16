# Wave 05 - faithful iGSM transfer

## Experiments

- Official facebookresearch/iGSM adapter and validator.
- Deterministic symbolic and non-symbolic recipe transfer.
- Token, sentence, and sentence+latent baselines.
- Add-backs of LDAD, residual prediction, rollout, hierarchy,
  counterfactual outcomes, and scalar geometry distillation.

Raw families: `real_*`, `lm_intent_faithful*`, and
`sentlm_*intent_faithful*`. Generated screen:
`runs/official_recipe_screen.md`.

## Completed one-seed matched baselines

| Model | Strict | +2 |
|---|---:|---:|
| Random | 0.200 | 0.460 |
| Token intent LM | 0.415 | 0.675 |
| Sentence intent LM | 0.450 | 0.755 |
| Sentence+latent likelihood | 0.455 | 0.765 |
| Sentence+latent distance | 0.445 | 0.715 |

Several deterministic transfer and add-back jobs were interrupted on
2026-07-14 when hierarchy became the sole active stage. Their partial logs and
checkpoints are retained and are not interpreted as results.

## Conclusion

Faithful iGSM will be the paper-facing domain, but the hierarchy and planner
must be selected before the official recipe matrix resumes.
