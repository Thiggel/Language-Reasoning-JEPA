# Wave 01 - representation and one-step dynamics

## Experiments

Run families: `disc_base*`, `disc_no_delta*`, `disc_chunkpred*`,
`disc_combo*`, `disc_nonres*`, `disc_novic*`, `disc_noema*`, data-scale
cells, frozen-encoder controls, and counterfactual transition audits.

## Principal results

| Intervention | Strict / +2 | Main diagnostic |
|---|---:|---|
| Pure/base JEPA | 0.350 / 0.750 | encoder-predictor collusion |
| Remove hybrid displacement | 0.150 / 0.510 | transition grounding weakens |
| Add frozen outcome anchor | 0.620 / 0.880 | terminal answer probe ~0.98 |
| Early combined model | 0.635 / 0.890 | matching and content both healthy |
| Replace cumulative state target by next observation | 0.245 / - | matching 0.22, rejected |

Counterfactual next-state matching reaches approximately 0.99-1.00 when the
dynamics are grounded and falls to chance after action shuffling. Direct
prediction is at least as strong as a residual skip in the controlled cells.

## Conclusion

Cumulative discourse-state prediction is necessary. A fixed outcome-space
anchor prevents collusion, and action-conditioned transition matching can be
audited independently of planning quality.
