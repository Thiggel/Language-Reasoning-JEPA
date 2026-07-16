# Wave 04 - preference horizon, search, and historical hierarchy

## Experiments

- GAR teacher horizon H=1/2/4/8/16 with greedy continuation.
- Bounded teacher beam width B=1/2/4/8.
- Root alternative count K=2/4/8.
- Historical K=3 shared-latent macro-transition auxiliary.
- Historical flat versus macro lookahead using oracle future-action
  enumeration.

Generated table: `runs/preference_sweep.md`.

## Results

| H, B=1 | Teacher top-1 | Strict |
|---:|---:|---:|
| 1 | 0.750 | 0.315 |
| 2 | 0.930 | 0.750 |
| 4 | 0.900 | 0.695 |
| 8 | 0.830 | 0.620 |
| 16 | 0.790 | 0.655 |

At H=4, increasing B from 1/4/8 improves teacher top-1 from
0.810/0.870/0.900, but the end-to-end B=8 students were interrupted under the
hierarchy-only directive. Removing the historical macro-transition training
loss changed 0.750/0.940 to 0.765/0.940.

## Conclusion

Longer greedy GAR is not monotonically better. The historical hierarchy
result tested a training auxiliary and oracle-action macro lookahead, not a
deployable top-down hierarchical planner. Wave 08 replaces it.
