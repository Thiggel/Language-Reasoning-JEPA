# Decision-question backlog

Order by expected decision value divided by elapsed time, not by novelty.

| Priority | Concrete question | Decision changed by answer | Smallest faithful test |
|---|---|---|---|
| 1 | Do hierarchy-health measurements detect known collapse, leakage, and healthy prediction on tiny synthetic controls? | Whether any further hierarchy comparison is interpretable. | Seconds-scale CPU/GPU fixtures plus one tiny train/eval run. |
| 2 | Can text-only span targets improve future-state prediction beyond an information-matched flat predictor without boundary leakage? | Whether to continue fixed/multi-scale span hierarchy. | Small fixed-token-span screen with shuffled-target and flat controls. |
| 3 | Which stabilizer ranges keep each hierarchy level non-collapsed under matched compute? | Whether VICReg, SIGReg, or another mechanism enters the main recipe. | Separate coarse log-scale ranges, then local refinement only for viable regions. |
| 4 | Are trajectories, batch diversity, and data mixture sufficient for 10/100-token transitions? | Whether failure is architectural or caused by sampling/packing. | Audit distributions before training; controlled packing comparison if deficient. |
| 5 | Does a discovered higher-level representation improve generation or planning rather than only linear probes? | Whether to scale toward paper training. | Frozen-encoder matched downstream evaluation with a flat control. |
| 1 (sequence edit) | How many unique oracle-denoising trajectories and exact same-state alternative outcomes are needed for healthy primitive edit dynamics? | Select the data anchor and whether counterfactual outcome prediction enters later edit experiments. | K=0 data curve at 512/2k/6k plus K={0,1,4,8} at 2k, one seed. |
| 2 (sequence edit) | Do unique problems or repeated transitions drive the data effect, and can exact alternatives substitute for unique anchors? | Whether to spend compute on diversity or local action coverage. | Fixed 18k-example exposure at 2k×9, 6k×3, 18k×1 crossed with selected K. |
| 3 (sequence edit) | After primitive validity passes, do hierarchy, dense rollout, or LDAD improve recursive dynamics under matched information and compute? | Which architectural ideas deserve confirmation. | Flat/H4/H8, H4 dense, and H4 no-LDAD using the selected data/K recipe. |
| 1 (sequence edit, active) | Does exact changed-step outcome supervision make the predictor use the edit action? | Whether the global-state edit formulation can be repaired or must be redesigned. | K=4 at 2k with local-target weights .25/1/4 and a high-weight LR cross-check. |
