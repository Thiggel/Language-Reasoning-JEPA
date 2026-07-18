# Experiment index

See `research/intent_phrase/README.md`, its wave documents, and controller rounds classified as `intent_phrase`.

- `2026-07-17-j3-gap-audit`: three-seed artifact-only localization; teacher
  and student rankings are strong, predicted/rollout task information is weak;
  causal-context and learning-rate diagnostic selected.
- `2026-07-17-j3-context-optimization-screen`: four jobs submitted from exact
  commit; infrastructure-invalid before optimization because the controller
  overwrote plan-level `TMPDIR`; corrected v2 awaits slot termination.
- `2026-07-17-j3-learning-rate-screen`: valid completed/recovery artifacts
  select full-history `1e-3` at seed 0 (`.705` strict); context 1, context 4,
  and `1e-4` regress. Seeds 1 and 2 are the sole next confirmation.

- `2026-07-16-gar-geometry-screen`: seven first-launch jobs terminal but
  infrastructure-invalid (`AF_UNIX path too long`); zero scientific results;
  valid retry/current work unresolved.
- `2026-07-16-paper-causal-counterfactual-repair`: seeds 1 and 2 failed before
  training because the immutable snapshot lacked `paper_causal_a_cfout`; seed
  0 has no terminal summary; zero scientific results.
- `2026-07-16-gar-h4-recovery`: planned single-cell infrastructure-safe retry
  of `H=4, K=2`, seed 0, with a short temporary path; compare only against the
  existing matched `H=2, K=2` seed.
