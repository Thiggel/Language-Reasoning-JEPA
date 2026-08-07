"""Goal/value heads scoring latent states against the prompt."""

from __future__ import annotations

import math

import torch
from torch import nn

from textjepa.models.layers import encoder_stack, mlp


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


class DirectActionRankHead(nn.Module):
    """Information-matched scorer that bypasses successor prediction.

    This is the behavior-cloning control for GAR: it sees the current history
    state, prompt/goal state, and candidate action code, but never ``F(s,a)``.
    Its hidden width matches :class:`ValueHead` closely in parameter count.
    """

    def __init__(self, d_state: int, d_action: int, hidden_mult: int = 2):
        super().__init__()
        width = 2 * d_state + d_action
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            mlp([width, d_state * hidden_mult], 1),
        )

    def forward(
        self, state: torch.Tensor, initial: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        initial = (
            initial.unsqueeze(-2).expand_as(state)
            if state.dim() > initial.dim() else initial
        )
        return self.net(torch.cat([state, initial, action], dim=-1)).squeeze(-1)


class TDQHead(nn.Module):
    """SARSA-style Q(z, u(a), z_0) baseline head (TD-JEPA-adapted).

    Higher Q is better internally (Q approximates -(discounted steps-to-go));
    planning exposes cost = -Q so the lower-is-better convention holds.
    """

    def __init__(self, d_state: int, d_action: int, hidden_mult: int = 2):
        super().__init__()
        width = 2 * d_state + d_action
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            mlp([width, d_state * hidden_mult], 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, initial: torch.Tensor
    ) -> torch.Tensor:
        while initial.dim() < state.dim():
            initial = initial.unsqueeze(-2)
        initial = initial.expand_as(state)
        return self.net(torch.cat([state, action, initial], dim=-1)).squeeze(-1)


class ExpectileValueHead(nn.Module):
    """Goal-value baseline V(z, z_0) = -||f(z) - g(z_0)||_2 (arXiv:2601.00844).

    ``f`` and ``g`` are small two-layer projections onto a d_state/2 metric
    space, so V <= 0 with V = 0 attainable exactly on goal-matching states.
    """

    def __init__(self, d_state: int, hidden_mult: int = 2):
        super().__init__()
        half = max(1, d_state // 2)
        self.f = nn.Sequential(
            nn.LayerNorm(d_state),
            mlp([d_state, d_state * hidden_mult], half),
        )
        self.g = nn.Sequential(
            nn.LayerNorm(d_state),
            mlp([d_state, d_state * hidden_mult], half),
        )

    def forward(self, state: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        while initial.dim() < state.dim():
            initial = initial.unsqueeze(-2)
        initial = initial.expand_as(state)
        return -torch.linalg.vector_norm(
            self.f(state) - self.g(initial), dim=-1
        )


class StateFeatureHead(nn.Module):
    """psi(z) -> R^{d_psi} state features for TD-JEPA successor prediction.

    Following the faithful TD-JEPA loss (Bagatella et al., arXiv:2510.00739,
    as specified for this repo) the psi term appears only under stop-gradient,
    so this head stays at its random initialization: successor features of a
    fixed random feature map are still well-defined, and the task-reward
    projection z_r is regressed against exactly these features.
    """

    def __init__(self, d_state: int, d_psi: int = 32, hidden_mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_state),
            mlp([d_state, d_state * hidden_mult], d_psi),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TaskEmbeddingHead(nn.Module):
    """tau(z_0) -> R^{d_task} task embedding for TD-JEPA.

    Adaptation choice: in the intent-phrase environment the task is fully
    specified by the problem statement, which ``z_0`` encodes, so the task
    embedding is a small MLP of the prompt state instead of a separately
    sampled task latent.
    """

    def __init__(self, d_state: int, d_task: int = 32, hidden_mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_state),
            mlp([d_state, d_state * hidden_mult], d_task),
        )

    def forward(self, initial: torch.Tensor) -> torch.Tensor:
        return self.net(initial)


class SuccessorFeatureHead(nn.Module):
    """T(z, u(a), z_task) -> R^{d_psi} successor features (TD-JEPA).

    Predicts the discounted sum of future ``psi`` state features under the
    demonstrated policy; Q(s, a) is recovered as ``T(...)^T z_r`` with the
    task-reward projection ``z_r`` regressed from rewards on training traces.
    """

    def __init__(
        self,
        d_state: int,
        d_action: int,
        d_task: int,
        d_psi: int = 32,
        hidden_mult: int = 2,
    ):
        super().__init__()
        width = d_state + d_action + d_task
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            mlp([width, d_state * hidden_mult], d_psi),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, task: torch.Tensor
    ) -> torch.Tensor:
        while task.dim() < state.dim():
            task = task.unsqueeze(-2)
        task = task.expand(*state.shape[:-1], task.shape[-1])
        return self.net(torch.cat([state, action, task], dim=-1))


def ridge_reward_projection(
    features: torch.Tensor, rewards: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    """z_r = argmin_z sum_s (psi(s)^T z - r(s))^2 + eps * ||z||^2.

    ``features`` is [N, d_psi], ``rewards`` is [N]; returns [d_psi].
    """
    d_psi = features.shape[-1]
    gram = features.T @ features + eps * torch.eye(
        d_psi, dtype=features.dtype, device=features.device
    )
    return torch.linalg.solve(gram, features.T @ rewards)


class GoalHead(nn.Module):
    """g(z_0) -> predicted terminal goal state (Takai et al., JSAI 2026).

    Adaptation choice: Takai et al. condition on a language instruction; in
    the intent-phrase environment the instruction is the problem statement,
    already encoded in ``z_0``, so the head is a two-layer MLP of the prompt
    state.  Trained toward the EMA-encoded solved-trajectory endpoint
    (training only); planning scores imagined endpoints by latent distance
    to this prediction — the non-oracle counterpart of ``energy=oracle_goal``.
    """

    def __init__(self, d_state: int, hidden_mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_state),
            mlp([d_state, d_state * hidden_mult], d_state),
        )

    def forward(self, initial: torch.Tensor) -> torch.Tensor:
        return self.net(initial)


class TransitionEnergyHead(nn.Module):
    """Lower-is-better Energy of a predicted state transition."""

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
        successor: torch.Tensor,
        initial: torch.Tensor,
    ) -> torch.Tensor:
        while initial.dim() < state.dim():
            initial = initial.unsqueeze(-2)
        initial = initial.expand_as(state)
        return self.net(
            torch.cat([state, successor, initial], dim=-1)
        ).squeeze(-1)


class HorizonEnergyHead(nn.Module):
    """Lower-is-better Energy of an endpoint relative to one MPC root.

    Unlike :class:`TransitionEnergyHead`, ``root`` remains fixed across every
    beam compared by one MPC decision.  A scalar horizon input lets one head
    score endpoints reached after different amounts of imagination without
    treating rollout depth as an unobserved nuisance variable.
    """

    def __init__(
        self, d_state: int, hidden_mult: int = 2, use_horizon: bool = True
    ):
        super().__init__()
        width = 3 * d_state + 1
        self.use_horizon = use_horizon
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            mlp([width, d_state * hidden_mult], 1),
        )

    def forward(
        self,
        root: torch.Tensor,
        endpoint: torch.Tensor,
        initial: torch.Tensor,
        horizon: torch.Tensor | float | int,
    ) -> torch.Tensor:
        while initial.dim() < root.dim():
            initial = initial.unsqueeze(-2)
        initial = initial.expand_as(root)
        horizon = torch.as_tensor(
            horizon, dtype=root.dtype, device=root.device
        )
        while horizon.dim() < root.dim() - 1:
            horizon = horizon.unsqueeze(-1)
        horizon = horizon.expand(root.shape[:-1]).unsqueeze(-1)
        # log1p keeps depths 1--16 on a modest, monotone numerical scale.
        horizon = torch.log1p(horizon) / 4.0
        if not self.use_horizon:
            # Horizon-blind control: one shared Energy for every depth.
            horizon = torch.zeros_like(horizon)
        return self.net(
            torch.cat([root, endpoint, initial, horizon], dim=-1)
        ).squeeze(-1)


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


class GaussianActionPrior(nn.Module):
    """Learned prior p(a | s) over intent-phrase action embeddings.

    An MLP maps the current state latent to the parameters of a (mixture of)
    isotropic Gaussian(s) over the action-embedding space: per component a
    mixture logit, a mean vector, and one scalar log-variance. Trained with
    the NLL of the observed next action's embedding, it lets the planner
    rank catalogue actions without any feasibility oracle.
    """

    def __init__(
        self,
        d_state: int,
        d_action: int,
        hidden: int = 256,
        n_components: int = 1,
    ):
        super().__init__()
        if n_components < 1:
            raise ValueError("action prior requires at least one component")
        self.d_action = d_action
        self.n_components = n_components
        self.net = nn.Sequential(
            nn.LayerNorm(d_state),
            mlp([d_state, hidden], n_components * (2 + d_action)),
        )

    def components(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[.., d_state] -> mixture logits [.., K], means [.., K, d_action],
        scalar (isotropic) log-variances [.., K]."""
        raw = self.net(state).reshape(
            *state.shape[:-1], self.n_components, 2 + self.d_action
        )
        logits = raw[..., 0]
        mu = raw[..., 1 : 1 + self.d_action]
        logvar = raw[..., 1 + self.d_action].clamp(-6.0, 4.0)
        return logits, mu, logvar

    def log_prob(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """log p(action | state) for matching leading shapes [.., d_*]."""
        logits, mu, logvar = self.components(state)
        diff = action.unsqueeze(-2) - mu
        component_lp = -0.5 * (
            self.d_action * (logvar + math.log(2.0 * math.pi))
            + diff.square().sum(-1) * (-logvar).exp()
        )
        return torch.logsumexp(logits.log_softmax(-1) + component_lp, dim=-1)

    def nll(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(state, action)


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


class TokenHistoryActionSupportHead(nn.Module):
    """Availability scorer that preserves lexical tokens in action history.

    Candidate tokens query every token in the already executed or imagined
    intent prefix.  Inputs are frozen token embeddings; only this small head
    learns.  The interface contains no symbolic dependency labels or future
    feasible-action menu.
    """

    def __init__(
        self,
        d_state: int,
        d_token: int,
        n_heads: int = 4,
        hidden_mult: int = 2,
        use_history: bool = True,
        max_action_tokens: int = 64,
    ):
        super().__init__()
        if d_token % n_heads:
            raise ValueError("token history attention heads must divide d_token")
        self.use_history = bool(use_history)
        self.n_heads = int(n_heads)
        self.head_width = d_token // n_heads
        self.token_position = nn.Parameter(
            torch.zeros(1, max_action_tokens, d_token)
        )
        nn.init.normal_(self.token_position, std=0.02)
        self.candidate_encoder = encoder_stack(
            d_token, 1, n_heads, hidden_mult, dropout=0.0
        )
        self.query = nn.Linear(d_token, d_token)
        self.key = nn.Linear(d_token, d_token)
        self.value = nn.Linear(d_token, d_token)
        self.attention_out = nn.Linear(d_token, d_token)
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_token))
        width = d_state + 3 * d_token
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            mlp([width, d_state * hidden_mult], 1),
        )

    @staticmethod
    def _token_mask(embeddings: torch.Tensor) -> torch.Tensor:
        # Padding embeddings are explicitly zeroed by DiscourseJEPA and the
        # planner.  Learned non-padding embeddings are not constrained to zero.
        return embeddings.detach().abs().sum(dim=-1).gt(0)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action.dim() < 3:
            raise ValueError("token-history actions require [..., tokens, width]")
        leading = state.shape[:-1]
        n = state.reshape(-1, state.shape[-1]).shape[0]
        length, width = action.shape[-2:]
        candidate = action.reshape(n, 1, length, width)
        if history is None:
            history = action.new_zeros(*leading, 0, length, width)
        history = history.reshape(
            n, history.shape[-3], history.shape[-2], history.shape[-1]
        )
        if history_mask is not None:
            history_mask = history_mask.reshape(n, history.shape[1])
        scores = self.score_candidate_set(
            state.reshape(n, 1, state.shape[-1]),
            candidate,
            history,
            history_mask,
        )
        return scores[:, 0].reshape(leading)

    def score_candidate_set(
        self,
        state: torch.Tensor,
        candidate: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        encoded_candidate: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Score ``V`` candidates while sharing one history without copies.

        Shapes are state ``[B,V,S]``, candidate ``[B,V,L,D]``, and history
        ``[B,H,Lh,D]``.  Multi-head attention is evaluated with einsums so the
        history is never repeated across candidates.
        """
        B, V, length, width = candidate.shape
        if length > self.token_position.shape[1]:
            raise ValueError("action phrase exceeds token support position limit")
        if encoded_candidate is None:
            encoded, candidate_valid = self.encode_candidate_set(candidate)
        else:
            encoded, candidate_valid = encoded_candidate
            if encoded.shape != candidate.shape or candidate_valid.shape != candidate.shape[:-1]:
                raise ValueError("encoded candidate cache shape mismatch")

        h_count, h_length = history.shape[1:3]
        if h_length > self.token_position.shape[1]:
            raise ValueError("history phrase exceeds token support position limit")
        history_valid = self._token_mask(history)
        if history_mask is None:
            action_valid = torch.ones(
                B, h_count, dtype=torch.bool, device=candidate.device
            )
        else:
            action_valid = history_mask.bool()
        if not self.use_history:
            action_valid = torch.zeros_like(action_valid)
        history_valid &= action_valid.unsqueeze(-1)
        positioned_history = (
            history + self.token_position[:, :h_length].unsqueeze(1)
        ).reshape(B, h_count * h_length, width)
        flat_valid = history_valid.reshape(B, h_count * h_length)
        keys = torch.cat([
            self.null_history.expand(B, -1, -1), positioned_history
        ], dim=1)
        key_valid = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=candidate.device),
            flat_valid,
        ], dim=1)

        q = self.query(encoded).reshape(
            B, V, length, self.n_heads, self.head_width
        ).permute(0, 1, 3, 2, 4)
        k = self.key(keys).reshape(
            B, keys.shape[1], self.n_heads, self.head_width
        ).permute(0, 2, 1, 3)
        value = self.value(keys).reshape(
            B, keys.shape[1], self.n_heads, self.head_width
        ).permute(0, 2, 1, 3)
        attention = torch.einsum("bvhld,bhsd->bvhls", q, k)
        attention = attention / (self.head_width ** 0.5)
        attention = attention.masked_fill(
            ~key_valid[:, None, None, None, :], float("-inf")
        ).softmax(dim=-1)
        context = torch.einsum(
            "bvhls,bhsd->bvhld", attention, value
        ).permute(0, 1, 3, 2, 4).reshape(B, V, length, width)
        context = self.attention_out(context)
        keep = candidate_valid.unsqueeze(-1).to(encoded.dtype)
        denom = keep.sum(dim=2).clamp_min(1.0)
        candidate_pool = (encoded * keep).sum(dim=2) / denom
        context_pool = (context * keep).sum(dim=2) / denom
        match_pool = (encoded * context * keep).sum(dim=2) / denom
        features = torch.cat([
            state,
            candidate_pool,
            context_pool,
            match_pool,
        ], dim=-1)
        return self.net(features).squeeze(-1)

    def encode_candidate_set(
        self, candidate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, V, length, width = candidate.shape
        if length > self.token_position.shape[1]:
            raise ValueError("action phrase exceeds token support position limit")
        candidate_valid = self._token_mask(candidate)
        candidate_key_padding = ~candidate_valid.reshape(B * V, length)
        dead_candidate = ~candidate_valid.reshape(B * V, length).any(dim=-1)
        candidate_key_padding[dead_candidate, 0] = False
        encoded = self.candidate_encoder(
            candidate.reshape(B * V, length, width)
            + self.token_position[:, :length],
            src_key_padding_mask=candidate_key_padding,
        ).reshape(B, V, length, width)
        return encoded, candidate_valid


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
