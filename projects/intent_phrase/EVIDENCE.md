# Evidence

See `STATUS.md` for the current causal matrix. Accurate transition prediction alone does not yield strong action selection; latent-goal preference distillation is the largest supported gain. Hierarchy is excluded from the paper-facing claim.

The learned full-catalogue proposal interface now has a supported one-seed
pilot: token-level causal matching to executed intent history reaches `.175`
strict and `.717` slack-two success at length nine, versus best observed
phrase-pooled values `.083/.358`; its gain is prior-only and is not evidence
for JEPA reranking.
