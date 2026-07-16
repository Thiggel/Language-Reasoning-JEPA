# Wave 03 - non-symbolic action selection and reduced reference

## Experiments

- One-step geometry-based advantage ranking.
- Counterfactual outcome/transition augmentation at K=1/2/4/8.
- Clean GAR horizons H=1/2/4/8/16 and root alternatives K=2/4/8.
- Fixed-point component removal and matched add-back tests.
- Margin, label-gap, and distractor-exposure screens.

Exact generated tables: `runs/preference_sweep.md`, `runs/round2_screen.md`,
`runs/selector_screen.md`, and `runs/component_removal_matrix.md`.

## Principal results

| Model | Strict | +2 | Interpretation |
|---|---:|---:|---|
| One-step GAR pilot | 0.440 | 0.845 | too myopic |
| Clean CF+GAR H=2,K=2 pilot | 0.705 | 0.880 | non-symbolic ordering works |
| Reduced H=2,K=2, 3 seeds | 0.797 +/- 0.008 | 0.963 +/- 0.008 | current JEPA reference |
| Remove latent prediction | 0.495 | 0.825 | transition grounding lost |
| Remove geometric preference | 0.115 | 0.500 | selection lost |
| Remove VICReg | 0.615 | 0.755 | representation/robustness lost |
| Replace EMA by online stop-gradient | 0.720 | 0.910 | off-history recovery worsens |

The reduced model matches counterfactual transitions at 0.997. Its remaining
gap is local action ordering and recovery after an earlier distractor, not
transition identity.

## Conclusion

This wave supplies the provisional non-symbolic base for hierarchy studies.
Further shearing is stopped; future components will be selected in dedicated
scientific sweeps and assembled once.
