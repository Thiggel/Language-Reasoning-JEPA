# Evidence

See `STATUS.md` for the current causal matrix. Accurate transition prediction alone does not yield strong action selection; latent-goal preference distillation is the largest supported gain. Hierarchy is excluded from the paper-facing claim.

The learned full-catalogue proposal interface now has a supported one-seed
pilot: token-level causal matching to executed intent history reaches `.175`
strict and `.717` slack-two success at length nine, versus best observed
phrase-pooled values `.083/.358`; its gain is prior-only and is not evidence
for JEPA reranking. Human steering subsequently removed both the action prior
and feasibility prior from the paper-facing method; retain this result only as
historical evidence about a discarded hybrid.

E030 (2026-07-23, observed): exhaustive invalid-action consequences recover
some tiny-set training control but not full-catalogue held-out feasibility.
On eight held-out episodes, full-catalogue invalid-action rates remain roughly
88--94%.

E031 (2026-07-23, candidate-privileged diagnostic): unchanged aligned
checkpoints reach 87.5% ProofWriter and 50% PlanBench-3 success at +4 actions
when restricted to the current symbolic feasible subset, versus 0% and 12.5%
for their best full-catalogue counterparts. The PlanBench shuffled-action
control remains 0% feasible-only, supporting learned action grounding.
