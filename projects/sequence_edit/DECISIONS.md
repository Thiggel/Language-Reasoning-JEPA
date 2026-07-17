# Decisions

- Use literal token edits and official solution text.
- Exclude symbolic ranking and feasible-edit oracles.
- Treat failed pre-training runs as process failures, not negative evidence.
- Do not use curriculum VICReg weights 0.1 through 1.0 as the primitive default;
  carry 0.02 only as the low-cost LDAD combination anchor.
