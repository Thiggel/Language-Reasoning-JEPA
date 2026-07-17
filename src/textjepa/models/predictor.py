"""Action-conditioned latent predictors."""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.layers import mlp


class ActionConditionedPredictor(nn.Module):
    """s_hat_{t+1} = s_t + MLP([LN(s_t); a_t]) — residual latent dynamics."""

    def __init__(
        self,
        d_state: int,
        d_action: int,
        hidden_mult: int = 4,
        n_hidden_layers: int = 2,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.norm = nn.LayerNorm(d_state)
        dims = [d_state + d_action] + [d_state * hidden_mult] * n_hidden_layers
        self.net = mlp(dims, d_state)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([self.norm(s), a], dim=-1))
        return s + out if self.residual else out


class CausalHistoryPredictor(nn.Module):
    """Causal world model over state/action histories.

    Training receives teacher-forced states. Planning uses
    :meth:`rollout`, recursively inserting predicted waypoints while retaining
    the complete history. The same implementation is used for primitive and
    macro dynamics so a predictor never silently falls back to a Markov MLP.
    """

    causal_sequence = True

    def __init__(
        self,
        d_state: int,
        d_action: int,
        n_layers: int = 2,
        n_heads: int = 8,
        ff_mult: int = 4,
        max_steps: int = 64,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual
        self.inp = nn.Linear(d_state + d_action, d_state)
        self.pos = nn.Parameter(torch.zeros(1, max_steps, d_state))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_state,
            n_heads,
            d_state * ff_mult,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_state)
        self.out = nn.Linear(d_state, d_state)

    def _positions(self, length: int) -> torch.Tensor:
        if length <= self.pos.shape[1]:
            return self.pos[:, :length]
        return torch.nn.functional.interpolate(
            self.pos.transpose(1, 2),
            size=length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeeze = states.dim() == 2
        if squeeze:
            states, actions = states.unsqueeze(1), actions.unsqueeze(1)
        length = states.shape[1]
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=states.device),
            diagonal=1,
        )
        h = self.inp(torch.cat([states, actions], -1)) + self._positions(length)
        key_padding = None
        if valid is not None:
            key_padding = ~valid
            key_padding = key_padding.clone()
            key_padding[key_padding.all(1), 0] = False
        pred = self.out(self.norm(self.blocks(
            h, mask=causal, src_key_padding_mask=key_padding
        )))
        if self.residual:
            pred = states + pred
        return pred[:, 0] if squeeze else pred

    def rollout(
        self,
        start: torch.Tensor,
        codes: torch.Tensor,
        state_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressively predict while retaining an observed prefix.

        ``state_history`` contains ``s_0 ... s_t`` and ``action_history``
        contains ``a_0 ... a_{t-1}``. Each proposed action is appended before
        predicting the next state. Omitting both arguments starts a fresh
        rollout from ``start``.
        """
        n, horizon, _ = codes.shape
        if state_history is None:
            states = start.expand(n, -1).unsqueeze(1)
            actions = codes[:, :0]
        else:
            states = state_history
            actions = (
                codes[:, :0]
                if action_history is None
                else action_history
            )
            if states.shape[0] != n:
                states = states.expand(n, -1, -1)
            if actions.shape[0] != n:
                actions = actions.expand(n, -1, -1)
        predictions = []
        for step in range(horizon):
            actions = torch.cat([actions, codes[:, step:step + 1]], dim=1)
            cur = self.forward(states, actions)[:, -1]
            predictions.append(cur)
            states = torch.cat([states, cur.unsqueeze(1)], dim=1)
        return torch.stack(predictions, 1)


# Backward-compatible scientific name used by the hierarchy code and old
# checkpoints. Primitive and high-level predictors now share this class.
CausalMacroPredictor = CausalHistoryPredictor


class ProbabilisticActionConditionedPredictor(nn.Module):
    """Diagonal-Gaussian latent transition ``p(z' | z, u)``.

    The mean is residual by default; the variance is learned and bounded only
    for numerical stability.  Training evaluates the density of a sample from
    the EMA target distribution, as in the V-JEPA variational objective.
    """

    def __init__(
        self,
        d_state: int,
        d_action: int,
        hidden_mult: int = 4,
        n_hidden_layers: int = 2,
        residual: bool = True,
        min_logvar: float = -8.0,
        max_logvar: float = 3.0,
    ):
        super().__init__()
        self.residual = residual
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.norm = nn.LayerNorm(d_state)
        dims = [d_state + d_action] + [d_state * hidden_mult] * n_hidden_layers
        self.net = mlp(dims, 2 * d_state)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        mu_delta, logvar = self.net(
            torch.cat([self.norm(s), a], dim=-1)
        ).chunk(2, dim=-1)
        mu = s + mu_delta if self.residual else mu_delta
        return mu, logvar.clamp(self.min_logvar, self.max_logvar)


class FiLMPredictor(nn.Module):
    """Trunk-conditioned variant: the action modulates every hidden layer
    via FiLM (scale/shift), instead of one-shot input concatenation —
    the MLP analog of AdaLN trunk conditioning."""

    def __init__(
        self,
        d_state: int,
        d_action: int,
        hidden_mult: int = 4,
        n_hidden_layers: int = 2,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.norm = nn.LayerNorm(d_state)
        d_h = d_state * hidden_mult
        self.layers = nn.ModuleList()
        self.films = nn.ModuleList()
        d_in = d_state
        for _ in range(n_hidden_layers):
            self.layers.append(nn.Linear(d_in, d_h))
            self.films.append(nn.Linear(d_action, 2 * d_h))
            d_in = d_h
        self.out = nn.Linear(d_in, d_state)
        self.act = nn.GELU()

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = self.norm(s)
        for layer, film in zip(self.layers, self.films):
            gamma, beta = film(a).chunk(2, dim=-1)
            h = self.act((1 + gamma) * layer(h) + beta)
        out = self.out(h)
        return s + out if self.residual else out


class AttnEditPredictor(nn.Module):
    """Edit-track attention predictor: F(sentence_embs, mask, s, a).

    The pooled slot state cannot represent WHICH sentence an edit changes
    (audit matching stuck at 0.44); here the predictor cross-attends over
    the current buffer's per-sentence embeddings with action-conditioned
    queries, then outputs the next pooled state."""

    def __init__(self, d_state: int, d_action: int, n_heads: int = 4,
                 n_layers: int = 2, n_queries: int = 4):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, n_queries, d_state))
        nn.init.normal_(self.queries, std=0.02)
        self.a_proj = nn.Linear(d_action, d_state)
        self.s_proj = nn.Linear(d_state, d_state)
        layer = nn.TransformerDecoderLayer(
            d_state, n_heads, d_state * 4, 0.0, "gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, n_layers, norm=nn.LayerNorm(d_state)
        )
        self.out = nn.Linear(n_queries * d_state, d_state)

    def forward(
        self,
        sent_emb: torch.Tensor,   # [N, C, D] current buffer sentences
        sent_mask: torch.Tensor,  # [N, C]
        s: torch.Tensor,          # [N, D] pooled current state
        a: torch.Tensor,          # [N, d_action]
    ) -> torch.Tensor:
        q = self.queries + self.a_proj(a).unsqueeze(1) + self.s_proj(s).unsqueeze(1)
        key_pad = ~sent_mask
        key_pad = key_pad.clone()
        key_pad[key_pad.all(dim=-1), 0] = False
        h = self.decoder(q.expand(-1, self.queries.shape[1], -1)
                         if q.shape[1] == 1 else q, sent_emb,
                         memory_key_padding_mask=key_pad)
        return s + self.out(h.flatten(1))


class TokenAlignedEditPredictor(nn.Module):
    """Spatial edit transition over token-aligned latents.

    The location is a pointer into the *current* token sequence, not a textual
    absolute-position embedding.  Delete/insert/replace first construct the
    exact local latent scaffold; a bidirectional spatial Transformer then
    predicts the contextual next-token representations.  Rollout time remains
    causal because only the current predicted state and current action enter.
    """

    def __init__(self, d_state: int, d_action: int, n_layers: int = 2,
                 n_heads: int = 8, ff_mult: int = 4,
                 relative_radius: int = 32):
        super().__init__()
        self.d_action = d_action
        self.relative_radius = int(relative_radius)
        self.op = nn.Embedding(3, d_state)
        self.relative = nn.Embedding(2 * self.relative_radius + 3, d_state)
        self.action_code = nn.Sequential(
            nn.LayerNorm(4 * d_state),
            nn.Linear(4 * d_state, d_state), nn.GELU(),
            nn.Linear(d_state, d_action),
        )
        self.action_condition = nn.Linear(d_action, d_state)
        self.prompt_condition = nn.Linear(d_state, d_state)
        layer = nn.TransformerEncoderLayer(
            d_state, n_heads, d_state * ff_mult, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(
            layer, n_layers, norm=nn.LayerNorm(d_state),
            enable_nested_tensor=False,
        )
        self.out = nn.Linear(d_state, d_state)

    @staticmethod
    def _gather_context(states: torch.Tensor, mask: torch.Tensor,
                        positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, width, dim = states.shape
        lengths = mask.sum(-1).long()
        left_index = (positions - 1).clamp(min=0)
        right_index = positions.clamp(min=0)
        left_index = torch.minimum(left_index, (lengths - 1).clamp(min=0))
        right_index = torch.minimum(right_index, (lengths - 1).clamp(min=0))
        row = torch.arange(n, device=states.device)
        left = states[row, left_index]
        right = states[row, right_index]
        left = left * (lengths > 0).unsqueeze(-1)
        right = right * (lengths > 0).unsqueeze(-1)
        return left, right

    def encode_action(self, states: torch.Tensor, mask: torch.Tensor,
                      operations: torch.Tensor, positions: torch.Tensor,
                      content: torch.Tensor) -> torch.Tensor:
        left, right = self._gather_context(states, mask, positions)
        return self.action_code(torch.cat([
            self.op(operations.clamp(0, 2)), left, right, content,
        ], dim=-1))

    def _scaffold(self, states: torch.Tensor, mask: torch.Tensor,
                  operations: torch.Tensor, positions: torch.Tensor,
                  content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scaffold = states.new_zeros(states.shape)
        next_mask = torch.zeros_like(mask)
        width = states.shape[1]
        for row in range(states.shape[0]):
            length = int(mask[row].sum().item())
            op = int(operations[row].item())
            pos = int(positions[row].item())
            pos = max(0, min(pos, length if op == 1 else max(length - 1, 0)))
            current = states[row, :length]
            if op == 0:  # delete token at pointer
                edited = torch.cat([current[:pos], current[pos + 1:]], dim=0)
            elif op == 1:  # insert into the pointed gap
                edited = torch.cat([
                    current[:pos], content[row:row + 1], current[pos:]
                ], dim=0)
            else:  # replace pointed token
                edited = current.clone()
                if length:
                    edited[pos] = content[row]
            out_len = min(int(edited.shape[0]), width)
            scaffold[row, :out_len] = edited[:out_len]
            next_mask[row, :out_len] = True
        return scaffold, next_mask

    def forward(self, states: torch.Tensor, mask: torch.Tensor,
                operations: torch.Tensor, positions: torch.Tensor,
                content: torch.Tensor, prompt: torch.Tensor,
                return_action: bool = False):
        action = self.encode_action(
            states, mask, operations, positions, content
        )
        scaffold, next_mask = self._scaffold(
            states, mask, operations, positions, content
        )
        coordinate = torch.arange(states.shape[1], device=states.device)
        relative = coordinate.unsqueeze(0) - positions.unsqueeze(1)
        relative = relative.clamp(
            -self.relative_radius - 1, self.relative_radius + 1
        ) + self.relative_radius + 1
        h = scaffold + self.relative(relative)
        h = h + self.action_condition(action).unsqueeze(1)
        h = h + self.prompt_condition(prompt).unsqueeze(1)
        key_pad = ~next_mask
        key_pad = key_pad.clone()
        key_pad[key_pad.all(-1), 0] = False
        prediction = self.out(self.blocks(h, src_key_padding_mask=key_pad))
        prediction = prediction * next_mask.unsqueeze(-1)
        if return_action:
            return prediction, next_mask, action
        return prediction, next_mask
