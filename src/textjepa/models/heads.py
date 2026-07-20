"""Goal/value heads scoring latent states against the prompt."""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.layers import mlp


class ValueHead(nn.Module):
    """Predicts remaining necessary steps from (state, goal-state) pairs.

    Serves as the goal energy at planning time: lower predicted remaining
    steps = closer to the solution region for this prompt.
    """

    def __init__(self, d_state: int, hidden_mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d_state),
            mlp([2 * d_state, d_state * hidden_mult], 1),
        )

    def forward(self, s: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
        s0 = s0.unsqueeze(-2).expand_as(s) if s.dim() > s0.dim() else s0
        return self.net(torch.cat([s, s0], dim=-1)).squeeze(-1)


class MacroValueHead(nn.Module):
    """Cost/advantage head for a macro action in a problem state."""

    def __init__(
        self, d_state: int, d_macro: int, hidden_mult: int = 2
    ):
        super().__init__()
        width = 2 * d_state + d_macro
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            mlp([width, d_state * hidden_mult], 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        initial: torch.Tensor,
        macro: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([state, initial, macro], -1)).squeeze(-1)


class MacroSupportHead(nn.Module):
    """Conditional on-manifold score for state/macro pairs."""

    def __init__(
        self, d_state: int, d_macro: int, hidden_mult: int = 2
    ):
        super().__init__()
        width = d_state + d_macro
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            mlp([width, d_state * hidden_mult], 1),
        )

    def forward(
        self, state: torch.Tensor, macro: torch.Tensor
    ) -> torch.Tensor:
        return self.net(torch.cat([state, macro], -1)).squeeze(-1)


class ActionSupportHead(nn.Module):
    """Predict whether an intent-phrase action is available in a state."""

    def __init__(
        self, d_state: int, d_action: int, hidden_mult: int = 2
    ):
        super().__init__()
        width = d_state + d_action
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            mlp([width, d_state * hidden_mult], 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(torch.cat([state, action], -1)).squeeze(-1)


class HistoryActionSupportHead(nn.Module):
    """Score availability using an explicit, non-oracle action history.

    Candidate intent phrases query the intents already executed (or imagined
    within a beam). This exposes prerequisite identity without modifying the
    frozen JEPA state or consulting symbolic feasibility at inference.
    """

    def __init__(
        self,
        d_state: int,
        d_action: int,
        n_heads: int = 2,
        hidden_mult: int = 2,
        use_history: bool = True,
    ):
        super().__init__()
        if d_action % n_heads:
            raise ValueError("action history attention heads must divide d_action")
        self.use_history = bool(use_history)
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_action))
        self.attention = nn.MultiheadAttention(
            d_action, n_heads, batch_first=True
        )
        width = d_state + 2 * d_action
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            mlp([width, d_state * hidden_mult], 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        leading = state.shape[:-1]
        flat_action = action.reshape(-1, action.shape[-1])
        query = flat_action.unsqueeze(1)
        if history is None:
            history = action.new_zeros(*leading, 0, action.shape[-1])
        flat_history = history.reshape(
            flat_action.shape[0], history.shape[-2], history.shape[-1]
        )
        if history_mask is None:
            flat_mask = torch.ones(
                flat_history.shape[:2], dtype=torch.bool, device=action.device
            )
        else:
            flat_mask = history_mask.reshape(
                flat_action.shape[0], history_mask.shape[-1]
            ).bool()
        if not self.use_history:
            flat_mask = torch.zeros_like(flat_mask)
        null = self.null_history.expand(flat_history.shape[0], -1, -1)
        keys = torch.cat([null, flat_history], dim=1)
        key_padding = torch.cat([
            torch.zeros(
                flat_mask.shape[0], 1, dtype=torch.bool, device=action.device
            ),
            ~flat_mask,
        ], dim=1)
        context, _ = self.attention(
            query, keys, keys, key_padding_mask=key_padding,
            need_weights=False,
        )
        features = torch.cat([
            state.reshape(-1, state.shape[-1]),
            flat_action,
            context.squeeze(1),
        ], dim=-1)
        return self.net(features).reshape(leading)


class SubgoalActionHead(nn.Module):
    """Cost of a primitive action for reaching a latent subgoal."""

    def __init__(
        self, d_state: int, d_action: int, hidden_mult: int = 2
    ):
        super().__init__()
        width = 2 * d_state + d_action
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            mlp([width, d_state * hidden_mult], 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        subgoal: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            torch.cat([state, subgoal, action], -1)
        ).squeeze(-1)


class ControllerOutcomeHead(nn.Module):
    """Predict an outcome of closed-loop control toward a latent subgoal.

    Unlike an open-loop reachability metric, this head is trained on the state
    actually reached after the deployed lower controller replans for K steps.
    The initial prompt state is included because both task progress and the
    meaning of a discourse subgoal are problem-conditional.
    """

    def __init__(self, d_state: int, hidden_mult: int = 2):
        super().__init__()
        width = 3 * d_state
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            mlp([width, d_state * hidden_mult], 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        initial: torch.Tensor,
        subgoal: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            torch.cat([state, initial, subgoal], -1)
        ).squeeze(-1)
