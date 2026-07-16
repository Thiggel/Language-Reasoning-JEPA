# Wave 02 - geometry and symbolic action selection

## Experiments

- Temporal straightening weights 0.02/0.05/0.1.
- Straightening and monotonicity in raw versus projected latent geometry.
- State-distance and value monotonicity.
- Symbolic pairwise action ranking at K=1/2/4/8.
- Depth-calibrated cost ranking and scalar/value distillation.
- Early macro-transition and lookahead diagnostics.

Raw families: `disc_straight*`, `disc_proj_*`, `disc_mono*`, `disc_rank*`,
`disc_champion*`, and `disc_mdr*`.

## Principal results

| Model/intervention | Strict | +2 | Conclusion |
|---|---:|---:|---|
| Raw geometry, no straightening | 0.150 | 0.520 | geometry not a progress metric |
| Straightening 0.05 | 0.365 | 0.845 | raw-geometry sweet spot |
| Straightening 0.10 | 0.425 | 0.795 | content/value trade-off |
| Symbolic rank K=2 | 0.910 | 0.985 | strong but non-transferable |
| Symbolic preference, 3 seeds | 0.962 +/- 0.013 | 0.997 +/- 0.003 | diagnostic upper reference |

Historical deeper lookahead used the reference dependency graph to enumerate
future actions. It is now explicitly labeled oracle-action and is not evidence
for deployable deeper planning.

## Conclusion

Straightening reshapes geometry but competes with content. Symbolic ranking
solves selection but violates the desired supervision story, motivating GAR.
