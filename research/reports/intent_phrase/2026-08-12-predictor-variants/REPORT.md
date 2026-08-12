# Predictor variants: the recipe is not predictor-specific

_2026-08-12. Contract causal-falsifier/variant cells (task #7), stylized
iGSM 3-9 steps, LDAD headline recipe, lr 3e-4 seed 0, 300 episodes,
feasible-menu slack curves. Run dirs:
runs/autonomy/intent_phrase/2026-08-12-intent-predictor-variants-v1/._

## Background (real bug found on the way)

Horizon-mode GAR raised "requires the matched MLP predictor" for
causal-sequence predictors. Root cause was a semantic bug, not an
incompatibility: the rollout loop fed the causal predictor single-step
(state, action) pairs, so every imagined step re-entered the transformer at
position 0 with EMPTY history — silently degrading it to a Markov MLP.
Fixed (fa63cc9): rollout rows are grouped by anchor step, the factual
prefix is rebuilt, and predictor.rollout carries imagined history; a
bit-identity regression test proves the default MLP path unchanged.

## Results (strict / slack-4 by planning depth)

| depth | MLP headline (5-seed mean) | causal (s0) | non-residual MLP (s0) |
|---|---|---|---|
| 1  | .202 / —  | .327 / .927 | .157 / .913 |
| 2  | .729 / —  | .670 / .970 | .483 / .977 |
| 4  | .885 / —  | .850 / .990 | .930 / .997 |
| 8  | .965 / —  | .963 / 1.00 | .973 / 1.00 |
| 16 | .975 / —  | .970 / 1.00 | .980 / 1.00 |

Both variants land within noise of the headline at depth >= 8 (single seed
each). The causal predictor is stronger at depth 1-2 than non-residual;
non-residual catches up from depth 4. No variant collapses.

## Conclusion

Ablation-table entry: the endpoint-Energy planning recipe works across
predictor architectures (residual MLP, non-residual MLP, causal-sequence
with history); the mechanism is the energy/ranking supervision, not a
predictor idiosyncrasy. Single-seed variants; the plotted ablation will use
the long-trace retrain if promoted beyond the appendix.
