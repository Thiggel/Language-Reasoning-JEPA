# Decision-question backlog

Order by expected decision value divided by elapsed time, not by novelty.

| Priority | Concrete question | Decision changed by answer | Smallest faithful test |
|---|---|---|---|
| 1 | Do hierarchy-health measurements detect known collapse, leakage, and healthy prediction on tiny synthetic controls? | Whether any further hierarchy comparison is interpretable. | Seconds-scale CPU/GPU fixtures plus one tiny train/eval run. |
| 2 | Can text-only span targets improve future-state prediction beyond an information-matched flat predictor without boundary leakage? | Whether to continue fixed/multi-scale span hierarchy. | Small fixed-token-span screen with shuffled-target and flat controls. |
| 3 | Which stabilizer ranges keep each hierarchy level non-collapsed under matched compute? | Whether VICReg, SIGReg, or another mechanism enters the main recipe. | Separate coarse log-scale ranges, then local refinement only for viable regions. |
| 4 | Are trajectories, batch diversity, and data mixture sufficient for 10/100-token transitions? | Whether failure is architectural or caused by sampling/packing. | Audit distributions before training; controlled packing comparison if deficient. |
| 5 | Does a discovered higher-level representation improve generation or planning rather than only linear probes? | Whether to scale toward paper training. | Frozen-encoder matched downstream evaluation with a flat control. |

