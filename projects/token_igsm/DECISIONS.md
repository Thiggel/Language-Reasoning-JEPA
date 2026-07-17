# Decisions

- Keep fixed-span and semantic-boundary hierarchy as separate conditions.
- Treat oracle-goal planning as diagnostic.
- Require an information-matched flat control and executable low-level refinement.
- EMA targets must remain deterministic and in evaluation mode.
- Use learned value only at the topmost level until lower-level energies pass
  optimizer-exploitation tests; lower levels match selected subgoals by latent
  distance.
- Treat the token prior and conditional macro codebook as hard search support
  in the next diagnostic, not merely as soft penalties.
- Spend the remaining global headroom only on the missing seed-0 semantic flat
  control; do not expand seeds or scale until that matched comparison is valid.
- Never compare a primitive reached state directly with a distinct hierarchy
  state; lift the complete reached path through the corresponding EMA causal
  encoders first.
- Treat action-advantage targets as non-symbolic only when alternatives come
  from full-vocabulary token outcomes, observed chunks, or learned support—not
  symbolic feasibility filtering.
