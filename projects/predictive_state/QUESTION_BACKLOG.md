# Question backlog

0. Does any auxiliary pressure reach the backbone while an 8.8M-parameter
   predictor can solve the transition alone? Measured effect on the layer-24
   representation is 250x smaller than that of the co-trained next-token loss.
   Bottlenecking the predictor and raising `λ_pred` are the two direct tests,
   and both are cheaper than adding adaptation capacity.
1. How much of the conditional gain survives matched-action sampling?
2. Is the user's proposed Qwen 18/24 -> 12 split better than 6/12/18/24 depth
   fractions after parameter and learning-rate matching?
3. Does cosine plus scale outperform normalized MSE for recurrent injection?
4. How much history-window advantage remains after Stage 1?
5. Does rollout training improve the ordinary full path or only the jump path?
6. Is recurrent drift dominated by state direction, activation scale, upper KV
   cache error, or distribution shift in selected actions?
7. Does periodic refresh dominate pure recurrence at matched wall-clock cost?
8. Does learned distance add held-out branch-ranking signal beyond direct
   value, position, likelihood, and entropy?
9. Is a directional neural quasimetric needed after separate state/goal
   encoders, or does a whitened 128-dimensional Euclidean geometry suffice?
10. Does distance shaping improve verified success without reward-model
    exploitation?
