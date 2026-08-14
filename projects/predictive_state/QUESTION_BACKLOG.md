# Question backlog

Resolved and kept for the record: question 0 (auxiliary pressure does reach
the backbone; with the next-token loss off it moves to NLL 5.69, yet the
transition improves only slightly), and the compression hypothesis for the
residual transition error (refuted; error is flat across positions 0-1023).

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
11. Does jump drafting with exact speculative verification clear a useful
    speedup at the measured 0.52 top-1 acceptance rate? It is lossless by
    construction, so the excess-NLL gate does not apply.
12. Does the Stage 2 fidelity gap close at the protocol's full 60M-token
    curriculum budget, or only the 9% the 7M-token run bought?
13. Can an energy head trained purely by ranking counterfactual
    continuations against the observed one recover what the superseded
    remaining-chunk regression was supposed to provide?
