# Competitor JEPA value/energy baselines: verified literature and adaptation contract

_2026-08-06. Web-verified citations for the baseline suite requested for the
ICLR paper. No experiments in this report._

## Verified methods

| Abbrev. | Actual paper | Family |
|---|---|---|
| TD-JEPA | Bagatella, Pirotta, Touati, Lazaric, Tirinzoni, "TD-JEPA: Latent-predictive Representations for Zero-Shot RL", arXiv:2510.00739 | (a) amortized TD value, no test-time rollout |
| Temporal-Distance JEPA | Bai & Xiong, arXiv:2607.25337 | (b) TD-shaped geometry (quasimetric distance = steps-to-go) |
| FF-JEPA | Masip et al., arXiv:2606.09311 | (c) subgoal proposal + rollout latent distance |
| LAGO | Barbeau et al., arXiv:2606.20627 | (c) language-conditioned subgoal latents + soft-min rollout cost |
| "Takai et al." | **No such paper found — do not cite.** Closest real method: Destrade, Bounou, Le Lidec, Ponce, LeCun, "Value-guided action planning with JEPA world models", arXiv:2601.00844 | (b) V(s,g) = -||E(s)-E(g)|| with IQL-style expectile TD |
| PLDM | Sobal et al., arXiv:2502.14819 | (c) rollout + latent distance to encoded goal |
| HWM | Zhang et al., arXiv:2604.03208 | (c) hierarchical, already studied and negative here |
| DINO-WM | Zhou et al., arXiv:2411.04983 | (c) frozen encoder + distance energy |
| TD-MPC2 | Hansen et al., arXiv:2310.16828 | non-JEPA reference; needs reward head |
| EB-JEPA | Terver et al., arXiv:2602.03604 | (c) energy = prediction error; library reference |

## Loss formulations (plain text)

- **TD-JEPA (a)**: state encoder phi, task encoder psi, predictor
  T(phi(s), a, z) approximating successor features;
  loss = E || T(phi(s),a,z) - sg[psi(s')] - gamma * sg[T(phi(s'), a', z)] ||^2;
  test-time score = T(z_t, a, z_task)^T z_task, no imagination.
- **Destrade-style value-guided (b)**: V(s,g) = -||E(s) - E(g)||_2 trained with
  expectile TD regression L_tau^2(-1[s != g] + gamma*sg[V(s',g)] - V(s,g)),
  tau,gamma near 1, alongside the JEPA prediction+VICReg losses; plan by
  minimizing embedding distance of imagined endpoint to goal.
- **Temporal-Distance JEPA (b)**: directed quasimetric d(z_i,z_j) regressed on
  (j - i) with SmoothL1 plus a negative-pair margin; plan by
  argmin over action sequences of d(z_hat_H, z_goal).
- **FF-JEPA (c)**: action-free subgoal predictor G(z_history) -> z_{t+H}
  (MSE or diffusion); rank rollouts by ||F_rollout - z_sg||; replan every H.
- **LAGO (c)**: subgoal head conditioned on instruction embedding and
  completion scalar rho; K subgoals; cost = sum_k lambda^k *
  (-tau * log sum_h exp(-MSE(z_hat_h, z_sg_k)/tau)).

## Adaptation notes for the intent-phrase environment

1. Family (c) with an *encoded terminal goal* is exactly our existing
   `energy=oracle_goal` diagnostic (100% success): PLDM/DINO-WM-style
   baselines are therefore already covered as the labeled oracle row.
   The non-oracle versions of family (c) require a learned subgoal/goal
   predictor (FF-JEPA's G or LAGO's language-conditioned head), which is
   implementable here because z_0 encodes the problem statement.
2. Family (a) and (b) plan **without any goal latent at test time** in our
   setting: (a) scores (z_t, a, task=z_0); (b) scores imagined endpoints by a
   learned distance/value conditioned on z_0. These are the faithful
   "competitor energies" for the headline comparison against our
   endpoint-ranking energy.
3. Our endpoint-ranking energy differs from both TD families in that its
   labels are relative (pairwise geometric preference), not bootstrapped and
   not absolute distances — this is the paper's contrast axis, and prior
   internal evidence (all absolute/regression heads collapse at depth) says
   the ranking form is what makes deep search work.
4. TD-MPC2 requires a dense reward head; in our reward-sparse environment the
   honest adaptation is terminal-only reward, which reduces its Q-head to a
   success predictor — state this as the reason it appears only as a
   discussion reference, or implement with terminal reward and label it.

## Planned baseline implementations (priority order)

1. `td_value` head: T(z, u(a), z_0) with bootstrapped TD loss on offline
   traces (family a).
2. `expectile_goal_value`: V(z, z_0) = -||proj(z) - proj(z_0-conditioned goal
   embedding)|| with expectile TD (family b, Destrade-corrected).
3. `quasimetric_distance`: directed d(z, z') regressed on step gaps
   (family b, Temporal-Distance JEPA).
4. `subgoal_proposer` G(z_t, z_0, rho): enables non-oracle family (c)
   (FF-JEPA/LAGO-style) rows.
