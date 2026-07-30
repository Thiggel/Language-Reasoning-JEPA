# Evidence

- Higher-level objectives alter representation and optimization diagnostics, but causal abstraction has not been established.
- Continuous latent planning can exploit off-support macro-actions.
- Support restriction and low-level reachability are therefore required diagnostics.
- No current result justifies a reliable hierarchical-planning headline.
- A candidate-privileged, single-seed Qwen3.5-0.8B token-oracle pilot completed
  across iGSM ID and four OOD splits. Exact endpoint planning sometimes
  improves over greedy/proposal baselines, but neither exact nor learned
  performance scales monotonically with horizon.
- At ID horizon 8, exact and learned next-token accuracy are both 0.375;
  exact/learned span accuracy is 0.375/0.125. At ID horizon 16, next-token
  accuracy is 0.750/0.625 and span accuracy is 0.375/0.125.
- Counterfactual and multistep additions reduce training losses but lower the
  ID horizon-8 learned next-token accuracy from 0.375 to 0.125 in this pilot.
  This does not support advancing the hierarchy.
- These measurements use eight roots per cell and await Alex seed replication.
  They are oracle/candidate-privileged diagnostics, not end-to-end solution
  accuracy.
