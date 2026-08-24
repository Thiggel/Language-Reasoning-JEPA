"""Trajectory-geometry regularizers.

TemporalStraightening (arXiv:2603.12231): maximize cosine similarity of
consecutive latent velocities v_t = s_{t+1} - s_t so that Euclidean latent
distance approximates geodesic (minimum-step) distance — the property that
makes raw-geometry goal-distance planning work. Label-free, applied to
online states.

GoalMonotonicity: hinge on the LN-L1 distance to the trace-terminal EMA
state.  The legacy mode uses necessary/distractor annotations; the clean
``label_free`` mode instead asks every observed transition to decrease the
distance and never reads those annotations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from textjepa.objectives.base import Objective, masked_mean


def _all_states(out) -> torch.Tensor:
    """States the geometry losses act on: the geo projection when the model
    has one (decoupled metric), else the raw states [s0; s_1..T]."""
    if "geo_states" in out.extras:
        return out.extras["geo_states"]
    return torch.cat([out.s0.unsqueeze(1), out.step_states], dim=1)


def velocity_cosines(out) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine of consecutive velocities [B, T-1] and its validity mask."""
    v = torch.diff(_all_states(out), dim=1)  # [B, T, D]
    cos = F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1)
    return cos, out.step_mask[:, 1:].float()


def goal_distances(out) -> torch.Tensor:
    """LN-L1 distance of every state (incl. s0) to its trace-terminal EMA
    target state; [B, T+1]."""
    B = out.s0.shape[0]
    last = out.step_mask.sum(dim=1).clamp(min=1) - 1
    tgt = out.extras.get("geo_states_tgt", out.step_states_tgt)
    goal = tgt[torch.arange(B, device=out.s0.device), last]
    ln = lambda x: F.layer_norm(x, x.shape[-1:])
    return (ln(_all_states(out)) - ln(goal).unsqueeze(1)).abs().mean(-1)


class HindsightGoalMonotonicity(Objective):
    """GoalMonotonicity with the goal RELABELED to a random observed future.

    :class:`GoalMonotonicity` uses one goal per trajectory -- the terminal
    state -- so it supplies O(T) constraints and only ever describes
    "distance to done".  Hindsight relabeling (Andrychowicz et al., 2017)
    instead treats ANY observed future state ``s_j`` as a goal for the prefix
    before it: the trajectory demonstrably reached ``s_j``, so ``s_t -> s_j``
    is a true reachable-in-(j-t)-steps pair that needs no extra data and no
    symbolic label.  That turns the same trajectories into O(T^2) constraints
    and shapes the whole metric rather than one direction in it.

    Steps at or after the sampled goal index are masked out -- after reaching
    ``s_j`` there is nothing left to say about approaching it.
    """

    def __init__(self, margin: float = 0.02, n_goals: int = 2):
        super().__init__()
        self.margin = margin
        self.n_goals = int(n_goals)

    def forward(self, out, batch: dict) -> torch.Tensor:
        states = _all_states(out)  # [B, T+1, D]
        tgt = out.extras.get("geo_states_tgt", out.step_states_tgt)  # [B,T,D]
        mask = out.step_mask.float()
        B, T = mask.shape
        device = states.device
        ln = lambda x: F.layer_norm(x, x.shape[-1:])
        lens = mask.sum(1).clamp(min=1)
        bidx = torch.arange(B, device=device)
        steps = torch.arange(T, device=device).unsqueeze(0)
        total = states.new_zeros(())
        for _ in range(max(self.n_goals, 1)):
            # uniform goal index among this trajectory's real steps
            j = (torch.rand(B, device=device) * lens).long().clamp(max=T - 1)
            goal = tgt[bidx, j]                                   # [B, D]
            d = (ln(states) - ln(goal).unsqueeze(1)).abs().mean(-1)  # [B,T+1]
            delta = d[:, 1:] - d[:, :-1]                           # [B, T]
            valid = mask * (steps <= j.unsqueeze(1)).float()
            total = total + masked_mean(F.relu(delta + self.margin), valid)
        return total / max(self.n_goals, 1)


class EnergyMonotonicity(Objective):
    """The Energy of the observed trajectory must DECREASE step by step.

    Every existing Energy term compares candidates that share an identical
    root, initial state and horizon, so nothing ties energies at different
    timesteps to a common scale.  This is the missing cross-time constraint:
    from the single fixed root s_0, E(s_0, s_t, s_0, t) is required to fall as
    t grows, which puts every depth on one ruler.

    Self-supervised: it reads only the order of the observed trajectory, never
    a step count, a necessary/distractor annotation or any symbolic label.
    """

    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin

    def forward(self, out, batch: dict) -> torch.Tensor:
        e = out.extras.get("energy_monotone")
        if e is None:
            return out.s0.new_zeros(())
        delta = e[:, 1:] - e[:, :-1]  # want < 0
        return masked_mean(F.relu(delta + self.margin),
                           out.step_mask[:, 1:].float())


class TemporalStraightening(Objective):
    def forward(self, out, batch: dict) -> torch.Tensor:
        cos, mask = velocity_cosines(out)
        return masked_mean(1.0 - cos, mask)


class GoalMonotonicity(Objective):
    def __init__(
        self,
        margin: float = 0.02,
        distractor_weight: float = 0.5,
        label_free: bool = False,
    ):
        super().__init__()
        self.margin = margin
        self.distractor_weight = distractor_weight
        # label_free: assume every executed step is necessary (true for ~85%
        # of trace steps) — removes the necessary/distractor supervision
        self.label_free = label_free

    def forward(self, out, batch: dict) -> torch.Tensor:
        d = goal_distances(out)  # [B, T+1]
        delta = d[:, 1:] - d[:, :-1]  # <0 means the step moved toward goal
        # label_free must not touch batch["necessary"] at all -- the flat
        # pipeline does not even provide that key.
        nec = (
            torch.ones_like(delta)
            if self.label_free
            else batch["necessary"].float()
        )
        loss = nec * F.relu(delta + self.margin) + (
            self.distractor_weight * (1 - nec) * F.relu(-delta)
        )
        return masked_mean(loss, out.step_mask.float())
