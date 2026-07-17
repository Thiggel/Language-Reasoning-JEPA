"""Multiscale token-to-span JEPA for the action-free text project."""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.action import MacroActionModel
from textjepa.models.ema import EMATeacher
from textjepa.models.heads import ValueHead
from textjepa.models.layers import build_causal_attention_mask


class CausalTokenStateEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        max_len: int,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.n_heads = n_heads
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_model * ff_mult,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def _positions(self, length: int) -> torch.Tensor:
        if length <= self.pos.shape[1]:
            return self.pos[:, :length]
        return torch.nn.functional.interpolate(
            self.pos.transpose(1, 2),
            size=length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        valid = tokens.ne(self.pad_id)
        x = self.tok(tokens) + self._positions(length)
        mask = build_causal_attention_mask(valid, self.n_heads)
        return self.norm(self.blocks(x, mask=mask))


class CausalLatentPredictor(nn.Module):
    """Predict a complete shifted latent sequence with causal attention."""

    def __init__(
        self,
        d_state: int,
        d_action: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        max_len: int,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.inp = nn.Linear(d_state + d_action, d_state)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_state))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_state,
            n_heads,
            d_state * ff_mult,
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
        valid: torch.Tensor,
    ) -> torch.Tensor:
        length = states.shape[1]
        safe_valid = valid.clone()
        empty = ~safe_valid.any(1)
        safe_valid[empty, 0] = True
        x = self.inp(torch.cat([states, actions], -1)) + self._positions(length)
        mask = build_causal_attention_mask(safe_valid, self.n_heads)
        return self.out(self.norm(self.blocks(x, mask=mask)))


class TokenHierarchyJEPA(nn.Module):
    """Shared-latent token and fixed-span world models.

    Level 0 consumes one observed token action at a time. Level 1 concatenates
    the ordered projected lower-level actions in a fixed span and predicts the
    waypoint state in the same latent space.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        encoder_layers: int = 4,
        predictor_layers: int = 2,
        n_heads: int = 8,
        ff_mult: int = 4,
        max_len: int = 512,
        d_action: int = 64,
        macro_span: int = 8,
        d_macro: int = 32,
        macro_variational: bool = False,
        macro_encoder_kind: str = "concat",
        macro_concat_width: int = 8,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.macro_span = macro_span
        self.encoder = CausalTokenStateEncoder(
            vocab_size,
            pad_id,
            d_model,
            encoder_layers,
            n_heads,
            ff_mult,
            max_len,
        )
        self.teacher = EMATeacher(self.encoder)
        self.action = nn.Embedding(vocab_size, d_action, padding_idx=pad_id)
        self.low_predictor = CausalLatentPredictor(
            d_model,
            d_action,
            predictor_layers,
            n_heads,
            ff_mult,
            max_len,
        )
        self.macro = MacroActionModel(
            d_action,
            d_model,
            d_macro,
            macro_span,
            kind=macro_encoder_kind,
            variational=macro_variational,
            concat_width=macro_concat_width,
        )
        self.high_predictor = CausalLatentPredictor(
            d_model,
            d_macro,
            predictor_layers,
            n_heads,
            ff_mult,
            max(8, max_len // max(1, macro_span)),
        )
        self.high_value = ValueHead(d_model)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        self.teacher.update(self.encoder, momentum)

    def _reasoning_sequences(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        tokens: torch.Tensor,
        prompt_len: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, _, dim = states.shape
        lengths = tokens.ne(self.pad_id).sum(1)
        reasoning = (lengths - prompt_len).clamp_min(1)
        width = int(reasoning.max().item())
        prev = states.new_zeros(batch, width, dim)
        target = states.new_zeros(batch, width, dim)
        action_ids = tokens.new_full((batch, width), self.pad_id)
        valid = torch.zeros(batch, width, dtype=torch.bool, device=tokens.device)
        for b in range(batch):
            p = int(prompt_len[b].item())
            n = int(reasoning[b].item())
            prev[b, :n] = states[b, p - 1 : p - 1 + n]
            target[b, :n] = targets[b, p : p + n]
            action_ids[b, :n] = tokens[b, p : p + n]
            valid[b, :n] = True
        return {
            "prev": prev,
            "target": target,
            "action_ids": action_ids,
            "valid": valid,
            "lengths": reasoning,
        }

    def forward(
        self, tokens: torch.Tensor, prompt_len: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        states = self.encoder(tokens)
        with torch.no_grad():
            targets = self.teacher(tokens)
        seq = self._reasoning_sequences(states, targets, tokens, prompt_len)
        actions = self.action(seq["action_ids"])
        low_pred = self.low_predictor(seq["prev"], actions, seq["valid"])

        K = self.macro_span
        n_macro = torch.div(seq["lengths"], K, rounding_mode="floor")
        high_width = max(1, int(n_macro.max().item()))
        batch = tokens.shape[0]
        high_prev = states.new_zeros(batch, high_width, self.d_model)
        high_target = states.new_zeros(batch, high_width, self.d_model)
        windows = actions.new_zeros(batch, high_width, K, actions.shape[-1])
        high_valid = torch.zeros(
            batch, high_width, dtype=torch.bool, device=tokens.device
        )
        for b in range(batch):
            count = int(n_macro[b].item())
            for j in range(count):
                start = j * K
                high_prev[b, j] = seq["prev"][b, start]
                high_target[b, j] = seq["target"][b, start + K - 1]
                windows[b, j] = actions[b, start : start + K]
                high_valid[b, j] = True
        macro, macro_extras = self.macro.training_code(
            windows.reshape(batch * high_width, K, -1),
            high_prev.reshape(batch * high_width, -1),
        )
        macro = macro.reshape(batch, high_width, -1)
        macro_extras = {
            name: value.reshape(batch, high_width, *value.shape[1:])
            for name, value in macro_extras.items()
        }
        high_pred = self.high_predictor(high_prev, macro, high_valid)
        prompt_state = torch.stack(
            [states[b, int(prompt_len[b].item()) - 1] for b in range(batch)]
        )
        high_value = self.high_value(high_pred, prompt_state)
        final_target = torch.stack(
            [
                seq["target"][b, int(seq["lengths"][b].item()) - 1]
                for b in range(batch)
            ]
        )
        ln = lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:])
        high_value_target = (
            ln(high_target) - ln(final_target).unsqueeze(1)
        ).abs().mean(-1)
        return {
            **seq,
            **macro_extras,
            "states": states,
            "low_pred": low_pred,
            "actions": actions,
            "high_prev": high_prev,
            "high_target": high_target,
            "high_pred": high_pred,
            "high_valid": high_valid,
            "macro_codes": macro,
            "macro_action_windows": windows,
            "high_value": high_value,
            "high_value_target": high_value_target,
        }
