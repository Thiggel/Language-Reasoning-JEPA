# Evidence

See `STATUS.md` for the current causal matrix. Accurate transition prediction alone does not yield strong action selection; latent-goal preference distillation is the largest supported gain. Hierarchy is excluded from the paper-facing claim.

The three-seed J3 artifact audit finds a strong two-step teacher
(`.890 +/- .036` top-1 versus a privileged symbolic oracle) and strong student
ranking (`.907 +/- .006` top-1 versus oracle), but only `.588 +/- .013` strict
closed-loop success. Task-value decodability falls from 1.000 in observed
states to `.380 +/- .014` after one predicted transition and `.263 +/- .004`
under recursive rollout. State variance and effective rank are healthy. This
supports testing predictor fidelity and compounding deployment error before
increasing teacher horizon.

The 2026-07-16 GAR geometry launch is infrastructure-invalid: all seven jobs
timed out before optimization because the multiprocessing Unix-socket path was
too long. The external-cluster counterfactual repair seeds 1 and 2 are also
process failures because their immutable snapshot lacked the requested Hydra
configuration; seed 0 has no terminal summary. None of these jobs changes the
scientific evidence.

The easier stylized domain does not currently establish a JEPA-over-matched-LM
result: compact evidence reports `.827 +/- .003` strict success for the matched
token intent policy versus `.797 +/- .008` for the older reduced non-symbolic
JEPA, with the current causal J3 reference at `.588 +/- .013`.

The valid seed-0 J3 optimization screen favors full-history learning rate
`1e-3`: `.705` strict and `.920` slack-two versus `.590`/`.890` for the matched
`3e-4` seed. Context 1 (`.155`), context 4 (`.475`), and `1e-4` (`.295`) fail
the behavioral gate. The winner remains provisional until seeds 1 and 2.
