# Decisions

- Keep hierarchy out of the paper-facing project.
- Use shuffled, information-matched action menus.
- Label oracle, symbolic, and candidate-privileged references explicitly.
- Retire the unsubmitted four-step teacher recovery: existing J3 artifacts
  already show a healthy teacher, so the next screen isolates causal history
  and learning-rate effects on predicted-state fidelity.
- Do not claim that JEPA beats the matched token intent policy on the easier
  stylized domain; the compact strict-success comparison currently favors the
  token policy (`.827` versus `.797` for the older reduced JEPA).
- Treat the 2026-07-16 GAR and counterfactual terminal process failures as zero
  scientific evidence and admit no duplicate/new round until current work is
  resolved and budget is sufficient.
- With the evening 16 GPU-hour allocation, admit only a three-GPU-hour
  four-step GAR recovery; do not duplicate the active Alex counterfactual seed.
- The seed-0 causal-history/optimization screen selects full-history `1e-3`
  for seeds 1 and 2 only: `.705` strict versus `.590` matched reference,
  without collapse. Do not widen the screen before replication.
