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
| Takai et al. | **CORRECTED (user-provided): found.** Takai, Takahashi, Ishida, Suzuki, Matsuo, "Language-Conditioned Latent Planning without Goal Images in JEPA World Models", JSAI 2026, 1E4-OS-39b-04 | (c) learned GoalHead g(z_0, u_text) -> z_goal_hat supervised by encoded goal image (L2 + cosine); plan by CEM/MPC minimizing d(imagined endpoint, z_goal_hat). MetaWorld reach: .15 vs .52 goal-image oracle |
| (kept as extra family-b row) | Destrade et al., "Value-guided action planning with JEPA world models", arXiv:2601.00844 | (b) V(s,g) = -||E(s)-E(g)|| with IQL-style expectile TD (implemented as `expectile_value`) |
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

## Faithfulness upgrades required (2026-08-07, after user confirmation)

1. **TD-JEPA (arXiv:2510.00739) exact adaptation**: our `td_q` SARSA head is an
   honest TD-family representative but not the paper's parameterization.
   Faithful version: successor-feature predictor T(phi(s), u(a), z) with task
   embedding z and factorization Q(s,a) = T(phi(s),u(a),z_r)^T z_r; offline TD
   loss || T(phi(s),a,z) - sg[psi(s')] - gamma*sg[T(phi(s'),a',z)] ||^2 with
   psi a learned state-feature head; test-time z_r regressed from the sparse
   task reward r(s)=1[solved] on training traces. Keep `td_q` as the simple
   TD ablation row; add `td_jepa` as the faithful row.
2. **Takai et al. (JSAI 2026) adaptation**: GoalHead g(z_0) -> z_goal_hat
   trained with ||z_goal_hat - sg[z_goal_EMA]||^2 + lambda*(1 - cos), where
   z_goal_EMA is the EMA-encoded solved trajectory endpoint (training only);
   plan with the existing distance planner but scoring
   d(imagined endpoint, z_goal_hat). This is the non-oracle counterpart of
   our oracle-goal diagnostic (oracle goal = 100% success), and the natural
   family-(c) row without symbolic goal access. In our environment the
   "instruction" is the problem statement, already encoded in z_0.

## Implementation notes for the faithful rows (added 2026-08-07)

Both upgrades are implemented as score modes `td_jepa` and `goal_head`
(cell variants `baseline_td_jepa` / `baseline_goal_head`, objectives
`td_jepa` / `goal_head`, heads in `src/textjepa/models/heads.py`,
losses in `src/textjepa/objectives/td_baselines.py`). Adaptation decisions:

1. **td_jepa** — task embedding: our task is fully specified by the problem
   statement, which z_0 encodes, so z_task = tau(z_0) with a small MLP tau
   (`TaskEmbeddingHead`). d_psi = d_task = 32 (`model.td_jepa_d_psi/_d_task`).
   The specified loss stop-gradients psi entirely, so psi stays a fixed
   random feature map (documented on `StateFeatureHead`); T and tau train.
   Bootstrap targets follow the td_q EMA convention (EMA next states under
   no_grad, detached online action codes, terminal step drops the bootstrap).
   Reward convention for z_r: r(s) = -1 on non-solved states (incl. z_0) and
   0 at the solved terminal — the repo's steps-to-go convention, an affine
   shift of 1[solved]. z_r is ridge-regressed (eps 1e-4) over training-trace
   states at plan time (`DiscourseJEPA.fit_td_jepa_reward_projection`,
   invoked automatically by `scripts/plan.py`; stored in the
   `core.td_jepa_z_r` buffer) — chosen over a train-loop hook because it
   works for any checkpoint without touching training. Planner cost:
   -T(z_pre_final, u(a_final), tau(z_0))^T z_r with the td_q pre-final-state
   convention. No training-time ga_energy exists for this mode (Q needs z_r).
2. **goal_head** — g(z_0) is a 2-layer MLP (`GoalHead`); the training target
   is the same EMA-encoded solved-trajectory endpoint the geometry teacher
   uses (`out.step_states_tgt` at the last valid step). Loss = mean-squared
   error + 1.0 * (1 - cosine) against the stop-gradient target. Planner cost
   = LN-L1 distance of the imagined endpoint to g(z_0), exactly mirroring
   `energy=oracle_goal` with the predicted goal in place of the oracle
   encoding; the same distance is emitted as the ga_energy diagnostic.

## First results (added 2026-08-06 night, one seed, 300 episodes, identical protocol)

| Method | D1 | D2 | D4 | D8 | D16 |
|---|---:|---:|---:|---:|---:|
| TD-Q (SARSA, amortized) | **.190** | .210 | .240 | .223 | .220 |
| Expectile goal-value | .057 | .063 | .103 | .173 | .187 |
| Horizon-blind endpoint ranking (5 seeds) | .123 | .438 | .777 | .800 | .801 |

TD-Q slack-2 *decreases* with depth (.617 -> .470): the amortized value is
myopically stronger than the ranking Energy (D1 .190 vs .123) but cannot
exploit recursive imagination — deeper search actively hurts it. The
expectile distance-value stays near the random baseline (.053). Both confirm
the internal pattern: absolutely-calibrated scores collapse under imagined
inputs; relative ranking distillation is what survives depth.
