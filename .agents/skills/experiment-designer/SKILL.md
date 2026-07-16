---
name: experiment-designer
description: Design small, decision-relevant TextJEPA experiments with matched controls, fair per-method tuning, validity gates, scale gates, and bounded compute. Use for ablations, stabilizer or learning-rate sweeps, hierarchy comparisons, data/packing tests, pilot-to-scale decisions, or run-plan construction.
---

# Design a decisive experiment

Start from one falsifiable question and write the result patterns that would
change the decision before choosing runs. Use the smallest scale that preserves
the mechanism, data trajectory, masking, target construction, and evaluation
needed for the claim.

Challenge target leakage, collapse, effective rank, target predictability,
capacity, optimization, target-encoder dynamics, packing, trajectory length,
batch diversity, data mix, proposal order, and evaluation validity. Include an
information-matched flat control and a negative control whenever they can
distinguish the proposed mechanism.

Tune alternatives fairly. Choose a coarse log-scale coefficient range for each
stabilizer based on its own loss scale; refine only viable regions. Add a
learning-rate cross-check when the mechanism changes gradient magnitude or
optimization stability. Do not spend full seed budgets on obviously collapsed
cells.

Declare primary metrics, health diagnostics, seeds, exclusions, early-stop
rules, expected artifacts, GPU-hours, and scale-up conditions. Keep tightly
paired comparisons on compatible hardware/software. Express each job as an
argv array in the plan; use `{python}` and `{root}` placeholders. Never call a
scheduler from this skill.

