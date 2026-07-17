"""Edit-JEPA: latent world model over text-buffer states.

The buffer (a draft solution) is encoded into a fixed set of latent slots
via cross-attention — length-invariant, order-aware. Actions are span
edits (delete/insert/replace) encoded from intent phrases. The same
LatentDynamicsCore provides transitions, Delta-JEPA decoding, hierarchy,
and the value head (predicted defects remaining = distance to perfect).
"""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.action import ActionEncoder
from textjepa.models.core import LatentDynamicsCore
from textjepa.models.ema import EMATeacher
from textjepa.models.layers import TokenTransformer
from textjepa.models.predictor import AttnEditPredictor
from textjepa.models.outputs import JEPAOutputs


class SlotBufferEncoder(nn.Module):
    """K learned slots cross-attend to [prompt | buffer] chunk embeddings."""

    def __init__(
        self,
        d_model: int,
        n_slots: int = 4,
        n_layers: int = 2,
        n_heads: int = 4,
        max_buffer_len: int = 32,
    ):
        super().__init__()
        self.slots = nn.Parameter(torch.zeros(1, n_slots, d_model))
        nn.init.normal_(self.slots, std=0.02)
        self.buffer_pos = nn.Parameter(torch.zeros(1, max_buffer_len, d_model))
        nn.init.normal_(self.buffer_pos, std=0.02)
        self.segment = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.segment, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * 4, 0.0, "gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.out = nn.Sequential(
            nn.Linear(n_slots * d_model, d_model), nn.LayerNorm(d_model)
        )

    def forward(
        self,
        prompt_emb: torch.Tensor,  # [N, P, D]
        prompt_mask: torch.Tensor,  # [N, P]
        buffer_emb: torch.Tensor,  # [N, C, D]
        buffer_mask: torch.Tensor,  # [N, C]
    ) -> torch.Tensor:
        C = buffer_emb.shape[1]
        memory = torch.cat(
            [
                prompt_emb + self.segment[0],
                buffer_emb + self.segment[1] + self.buffer_pos[:, :C],
            ],
            dim=1,
        )
        key_pad = ~torch.cat([prompt_mask, buffer_mask], dim=1)
        key_pad = key_pad.clone()
        key_pad[key_pad.all(dim=-1), 0] = False
        slots = self.slots.expand(memory.shape[0], -1, -1)
        h = self.decoder(slots, memory, memory_key_padding_mask=key_pad)
        return self.out(h.flatten(1))


class EditJEPA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        chunk_layers: int = 2,
        chunk_heads: int = 4,
        slot_layers: int = 2,
        slot_heads: int = 4,
        n_slots: int = 4,
        ff_mult: int = 4,
        max_chunk_len: int = 48,
        max_buffer_len: int = 32,
        d_action: int = 16,
        fsq_levels: list[int] | None = None,
        predictor_hidden_mult: int = 4,
        predictor_layers: int = 2,
        predictor_heads: int = 8,
        n_ops: int = 3,
        macro_k: int = 3,
        d_macro: int = 8,
        value_detach: bool = True,
        dropout: float = 0.0,
        chunk_target: str = "none",  # "none" | "frozen" outcome anchor
        geo_proj: bool = False,  # geometry losses act on a learned projection
        attn_predictor: bool = False,  # F attends over buffer sentences
        predictor_residual: bool = False,
        predictor_kind: str = "causal",
        high_predictor_kind: str = "causal",
        dense_rollout_depth: int = 0,
        high_dense_rollout_depth: int = 0,
        observed_action_ldad: bool = False,
        ldad_decoder_layers: int = 2,
        ldad_max_len: int = 12,
    ):
        super().__init__()
        self.chunk_target = chunk_target
        self.chunk_encoder = TokenTransformer(
            vocab_size, pad_id, d_model, chunk_layers, chunk_heads,
            ff_mult, max_chunk_len, dropout,
        )
        self.buffer_encoder = SlotBufferEncoder(
            d_model, n_slots, slot_layers, slot_heads, max_buffer_len
        )
        self.action_encoder = ActionEncoder(d_model, d_action, fsq_levels=fsq_levels)
        self.core = LatentDynamicsCore(
            d_model=d_model,
            d_action=d_action,
            predictor_hidden_mult=predictor_hidden_mult,
            predictor_layers=predictor_layers,
            predictor_heads=predictor_heads,
            n_ops=n_ops,
            macro_k=macro_k,
            d_macro=d_macro,
            value_detach=value_detach,
            geo_proj=geo_proj,
            residual=predictor_residual,
            predictor_kind=predictor_kind,
            high_predictor_kind=high_predictor_kind,
            dense_rollout_depth=dense_rollout_depth,
            high_dense_rollout_depth=high_dense_rollout_depth,
        )
        if observed_action_ldad:
            from textjepa.models.delta_decoder import ObservedActionDecoder

            self.observed_action_decoder = ObservedActionDecoder(
                d_model, vocab_size, ldad_max_len,
                n_layers=ldad_decoder_layers, n_heads=chunk_heads,
            )
        else:
            self.observed_action_decoder = None
        self.attn_pred = (
            AttnEditPredictor(d_model, d_action) if attn_predictor else None
        )
        self.chunk_teacher = EMATeacher(self.chunk_encoder)
        self.buffer_teacher = EMATeacher(self.buffer_encoder)
        # frozen random-init copies: fixed outcome anchors (never updated)
        self.chunk_anchor = EMATeacher(self.chunk_encoder)
        self.buffer_anchor = EMATeacher(self.buffer_encoder)

    @property
    def predictor(self):
        return self.core.predictor

    @property
    def value_head(self):
        return self.core.value_head

    def _encoders(self, mode: str):
        return {
            "online": (self.chunk_encoder, self.buffer_encoder),
            "teacher": (self.chunk_teacher, self.buffer_teacher),
            "anchor": (self.chunk_anchor, self.buffer_anchor),
        }[mode]

    def encode_chunks(
        self, tokens: torch.Tensor, teacher: bool = False, mode: str | None = None
    ) -> torch.Tensor:
        B, C, L = tokens.shape
        enc, _ = self._encoders(mode or ("teacher" if teacher else "online"))
        return enc(tokens.reshape(B * C, L)).reshape(B, C, -1)

    def encode_buffers(
        self,
        prompt_tokens: torch.Tensor,  # [B, P, L]
        prompt_mask: torch.Tensor,  # [B, P]
        buffer_tokens: torch.Tensor,  # [B, S, C, L]
        buffer_mask: torch.Tensor,  # [B, S, C]
        teacher: bool = False,
        mode: str | None = None,
    ) -> torch.Tensor:
        """Returns [B, S, D] buffer-state latents."""
        mode = mode or ("teacher" if teacher else "online")
        B, S, C, L = buffer_tokens.shape
        prompt_emb = self.encode_chunks(prompt_tokens, mode=mode)
        buf_emb = self.encode_chunks(buffer_tokens.reshape(B * S, C, L), mode=mode)
        P = prompt_emb.shape[1]
        prompt_rep = prompt_emb.unsqueeze(1).expand(B, S, P, -1).reshape(B * S, P, -1)
        pmask_rep = prompt_mask.unsqueeze(1).expand(B, S, P).reshape(B * S, P)
        _, enc = self._encoders(mode)
        states = enc(prompt_rep, pmask_rep, buf_emb, buffer_mask.reshape(B * S, C))
        return states.reshape(B, S, -1)

    def encode_actions(self, action_tokens: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(self.encode_chunks(action_tokens))

    @torch.no_grad()
    def update_teachers(self, momentum: float) -> None:
        self.chunk_teacher.update(self.chunk_encoder, momentum)
        self.buffer_teacher.update(self.buffer_encoder, momentum)

    def forward(self, batch: dict) -> JEPAOutputs:
        states = self.encode_buffers(
            batch["prompt_tokens"], batch["prompt_mask"],
            batch["buffer_tokens"], batch["buffer_mask"],
        )
        with torch.no_grad():
            states_tgt = self.encode_buffers(
                batch["prompt_tokens"], batch["prompt_mask"],
                batch["buffer_tokens"], batch["buffer_mask"], teacher=True,
            )
            action_emb_tgt = self.encode_chunks(batch["action_tokens"], teacher=True)
            step_emb_tgt = None
            if self.chunk_target == "frozen":
                step_emb_tgt = self.encode_buffers(
                    batch["prompt_tokens"], batch["prompt_mask"],
                    batch["buffer_tokens"], batch["buffer_mask"], mode="anchor",
                )[:, 1:]
        actions = self.action_encoder(self.encode_chunks(batch["action_tokens"]))
        if "changed_tokens" in batch and self.chunk_target == "frozen":
            with torch.no_grad():
                B2, T2, L2 = batch["changed_tokens"].shape
                slot_tgt = self.chunk_anchor(
                    batch["changed_tokens"].reshape(B2 * T2, L2)
                ).reshape(B2, T2, -1)
            self._slot_tgt = slot_tgt
        else:
            self._slot_tgt = None
        alt_actions = None
        if "alt_tokens" in batch:
            B, T, K, L = batch["alt_tokens"].shape
            alt_actions = self.encode_actions(
                batch["alt_tokens"].reshape(B, T * K, L)
            ).reshape(B, T, K, -1)
        overrides = {}
        if self.attn_pred is not None:
            B, S, C, L = batch["buffer_tokens"].shape
            sent = self.encode_chunks(
                batch["buffer_tokens"][:, :-1].reshape(B * (S - 1), C, L)
            )
            smask = batch["buffer_mask"][:, :-1].reshape(B * (S - 1), C)
            prev = states[:, :-1].reshape(B * (S - 1), -1)
            T = S - 1
            preds = self.attn_pred(
                sent, smask, prev, actions.reshape(B * T, -1)
            ).reshape(B, T, -1)
            overrides["preds_override"] = preds
            if alt_actions is not None:
                K = alt_actions.shape[2]
                alt_preds = self.attn_pred(
                    sent.repeat_interleave(K, 0),
                    smask.repeat_interleave(K, 0),
                    prev.repeat_interleave(K, 0),
                    alt_actions.reshape(B * T * K, -1),
                ).reshape(B, T * K, -1)
                overrides["alt_preds_override"] = alt_preds
        out = self.core(
            states[:, 0], states[:, 1:], states_tgt[:, 1:], actions,
            action_emb_tgt, batch["step_mask"], step_emb_tgt=step_emb_tgt,
            alt_actions=alt_actions, **overrides,
        )
        if self.observed_action_decoder is not None:
            out.extras["observed_action_logits"] = self.observed_action_decoder(
                out.step_states - out.prev_states
            )
        if "alt_buffer_tokens" in batch and "alt_preds" in out.extras:
            # Mechanical counterfactuals carry exact post-edit buffers but no
            # target-relative quality label. Encode every alternative outcome
            # independently with the EMA target and supervise dynamics only.
            B, T, K, C, L = batch["alt_buffer_tokens"].shape
            with torch.no_grad():
                alt_targets = self.encode_buffers(
                    batch["prompt_tokens"],
                    batch["prompt_mask"],
                    batch["alt_buffer_tokens"].reshape(B, T * K, C, L),
                    batch["alt_buffer_mask"].reshape(B, T * K, C),
                    teacher=True,
                ).reshape(B, T, K, -1)
            out.extras["cf_chunk_pred"] = out.extras["alt_preds"]
            out.extras["cf_chunk_tgt"] = alt_targets
            out.extras["cf_valid"] = (
                batch["alt_valid"] & out.step_mask.unsqueeze(-1)
            )
        if self._slot_tgt is not None:
            out.extras["slot_pred"] = self.core.chunk_head(out.preds)
            out.extras["slot_tgt"] = self._slot_tgt
        return out
