"""Competitor value-energy baselines: bootstrapped TD losses on demo traces.

Both objectives consume the supervision tensors emitted by
``DiscourseJEPA._td_baseline_supervision``: predictions from online encoders,
bootstrap values from EMA-teacher states (already computed under no_grad),
a valid-step mask, and a terminal indicator marking the solving step.
Reward convention is steps-to-go: r = -1 per executed step, 0 at terminal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from textjepa.objectives.base import Objective, masked_mean


class TDQ(Objective):
    """SARSA-style TD(0): Q(z_t, u(a_t), z_0) -> -(discounted steps-to-go).

    target = -1 + gamma * sg[Q(z_{t+1}, u(a_{t+1}), z_0)] on interior steps
    and -1 on the terminal (solving) step, i.e. reward only, no bootstrap.
    """

    def __init__(self, gamma: float = 0.98):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "td_q_pred" not in out.extras:
            return out.step_states.sum() * 0.0
        bootstrap = out.extras["td_next_value"].detach()
        keep = (~out.extras["td_terminal"]).float()
        target = -1.0 + self.gamma * bootstrap * keep
        error = (out.extras["td_q_pred"] - target).square()
        return masked_mean(error, out.extras["td_valid"].float())


class ExpectileValueTD(Objective):
    """Expectile TD regression L_tau^2(delta) for V(z, z_0) (IQL-style).

    delta = -1 + gamma * sg[V(z_{t+1}, z_0)] - V(z_t, z_0); positive deltas
    are weighted tau and negative deltas 1 - tau, so tau > 0.5 pushes V
    toward an optimistic expectile of the demonstrated returns.
    """

    def __init__(self, gamma: float = 0.98, tau: float = 0.9):
        super().__init__()
        if not 0.0 < tau < 1.0:
            raise ValueError("expectile tau must lie strictly in (0, 1)")
        self.gamma = float(gamma)
        self.tau = float(tau)

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "expectile_value_pred" not in out.extras:
            return out.step_states.sum() * 0.0
        bootstrap = out.extras["td_next_value"].detach()
        keep = (~out.extras["td_terminal"]).float()
        target = -1.0 + self.gamma * bootstrap * keep
        delta = target - out.extras["expectile_value_pred"]
        weight = torch.where(delta > 0, self.tau, 1.0 - self.tau)
        return masked_mean(
            weight * delta.square(), out.extras["td_valid"].float()
        )


class TDJEPASuccessor(Objective):
    """Faithful TD-JEPA successor-feature TD loss (Bagatella et al.,
    arXiv:2510.00739, adapted to demonstrated intent-phrase traces).

    || T(z_t, u(a_t), tau(z_0)) - sg[psi(z_{t+1})]
       - gamma * sg[T(z_{t+1}, u(a_{t+1}), tau(z_0))] ||^2

    The bootstrapped term is dropped on the terminal (solving) step, mirroring
    the terminal handling of :class:`TDQ`.  The squared norm is averaged over
    the d_psi feature dimension (same optimum; keeps the loss scale comparable
    across d_psi choices).  Rewards never appear here: they enter only through
    the post-training ridge regression of the task-reward projection z_r
    (``DiscourseJEPA.fit_td_jepa_reward_projection``).
    """

    def __init__(self, gamma: float = 0.98):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "td_jepa_pred" not in out.extras:
            return out.step_states.sum() * 0.0
        keep = (~out.extras["td_terminal"]).float().unsqueeze(-1)
        target = (
            out.extras["td_jepa_next_features"].detach()
            + self.gamma * out.extras["td_jepa_next_pred"].detach() * keep
        )
        error = (out.extras["td_jepa_pred"] - target).square().mean(-1)
        return masked_mean(error, out.extras["td_valid"].float())


class GoalHeadDistill(Objective):
    """GoalHead loss (Takai et al., JSAI 2026, adapted): L2 + cosine to the
    EMA-encoded solved-trajectory endpoint.

    || g(z_0) - sg[z_goal_EMA] ||^2 + cos_weight * (1 - cos(g(z_0), sg[...])),
    with the squared norm averaged over the state dimension.
    """

    def __init__(self, cos_weight: float = 1.0):
        super().__init__()
        self.cos_weight = float(cos_weight)

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "goal_head_pred" not in out.extras:
            return out.step_states.sum() * 0.0
        pred = out.extras["goal_head_pred"]
        target = out.extras["goal_head_target"].detach()
        mse = (pred - target).square().mean(-1)
        cosine = F.cosine_similarity(pred, target, dim=-1)
        return (mse + self.cos_weight * (1.0 - cosine)).mean()
