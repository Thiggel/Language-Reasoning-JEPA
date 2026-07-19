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
from textjepa.models.predictor import AttnEditPredictor, TokenAlignedEditPredictor
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
        token_aligned: bool = False,
        token_predictor_layers: int = 2,
        token_relative_radius: int = 32,
        counterfactual_encode_chunk_states: int = 64,
        gar_horizon: int = 1,
        selected_k: int = 0,
        proposal_selection: str = "hard",
        gar_teacher: str = "latent_distance",
        ldad_decoder_layers: int = 2,
        ldad_max_len: int = 12,
    ):
        super().__init__()
        self.chunk_target = chunk_target
        self.token_aligned = bool(token_aligned)
        self.counterfactual_encode_chunk_states = max(
            1, int(counterfactual_encode_chunk_states)
        )
        self.gar_horizon = max(1, int(gar_horizon))
        self.selected_k = max(0, int(selected_k))
        self.proposal_selection = str(proposal_selection)
        self.gar_teacher = str(gar_teacher)
        if self.proposal_selection not in {
            "hard", "random", "positive_anchor"
        }:
            raise ValueError(
                f"unknown proposal_selection: {self.proposal_selection}"
            )
        if (
            self.proposal_selection == "positive_anchor"
            and self.gar_teacher != "token_edit_distance"
        ):
            raise ValueError(
                "proposal_selection=positive_anchor requires "
                "gar_teacher=token_edit_distance"
            )
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
        self.token_pred = (
            TokenAlignedEditPredictor(
                d_model, d_action, n_layers=token_predictor_layers,
                n_heads=predictor_heads,
                relative_radius=token_relative_radius,
            ) if self.token_aligned else None
        )
        self.gar_head = nn.Sequential(
            nn.LayerNorm(d_model + d_action),
            nn.Linear(d_model + d_action, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, 1),
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

    def encode_token_buffers(
        self, buffer_tokens: torch.Tensor, mode: str = "online"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack sentence-contextualized tokens into one ordered token state."""
        B, S, C, L = buffer_tokens.shape
        encoder, _ = self._encoders(mode)
        hidden, valid = encoder.module.forward_tokens(
            buffer_tokens.reshape(B * S * C, L)
        ) if isinstance(encoder, EMATeacher) else encoder.forward_tokens(
            buffer_tokens.reshape(B * S * C, L)
        )
        hidden = hidden.reshape(B * S, C * L, -1)
        valid = valid.reshape(B * S, C * L)
        width = max(int(valid.sum(-1).max().item()), 1)
        packed = hidden.new_zeros(B * S, width, hidden.shape[-1])
        packed_mask = torch.zeros(
            B * S, width, dtype=torch.bool, device=hidden.device
        )
        for row in range(B * S):
            values = hidden[row, valid[row]]
            packed[row, :values.shape[0]] = values
            packed_mask[row, :values.shape[0]] = True
        return packed.reshape(B, S, width, -1), packed_mask.reshape(B, S, width)

    def encode_token_buffers_chunked(
        self, buffer_tokens: torch.Tensor, mode: str = "online"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode many state buffers with bounded peak activation memory.

        Counterfactual breadth multiplies the number of mechanically executed
        target states but must not change the scientific batch or optimizer
        schedule.  Chunk only the deterministic encoder evaluation, then pad
        each chunk to the common packed-token width before concatenation.
        """
        B, S = buffer_tokens.shape[:2]
        states_per_slice = max(
            1, self.counterfactual_encode_chunk_states // max(B, 1)
        )
        if S <= states_per_slice:
            return self.encode_token_buffers(buffer_tokens, mode=mode)
        chunks = [
            self.encode_token_buffers(
                buffer_tokens[:, start:start + states_per_slice], mode=mode
            )
            for start in range(0, S, states_per_slice)
        ]
        width = max(states.shape[-2] for states, _ in chunks)
        states_out, masks_out = [], []
        for states, mask in chunks:
            if states.shape[-2] < width:
                states = torch.nn.functional.pad(
                    states, (0, 0, 0, width - states.shape[-2])
                )
                mask = torch.nn.functional.pad(
                    mask, (0, width - mask.shape[-1])
                )
            states_out.append(states)
            masks_out.append(mask)
        return torch.cat(states_out, dim=1), torch.cat(masks_out, dim=1)

    @staticmethod
    def _pool_tokens(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.unsqueeze(-1).to(states.dtype)
        return (states * weight).sum(-2) / weight.sum(-2).clamp_min(1.0)

    def _structured_transitions(self, batch: dict, prompt_emb: torch.Tensor,
                                token_states: torch.Tensor):
        B, S, W, D = token_states.shape
        token_mask = batch["structured_token_mask"]
        T = S - 1
        operations = batch["op"][:, :T]
        positions = batch["edit_position"][:, :T]
        content_ids = batch["edit_content_token"][:, :T]
        content = self.chunk_encoder.tok(content_ids)
        prompt = prompt_emb.unsqueeze(1).expand(B, T, D)
        pred, pred_mask, actions = self.token_pred(
            token_states[:, :-1].reshape(B * T, W, D),
            token_mask[:, :-1].reshape(B * T, W),
            operations.reshape(-1), positions.reshape(-1),
            content.reshape(B * T, D), prompt.reshape(B * T, D),
            return_action=True,
        )
        pred = pred.reshape(B, T, W, D)
        pred_mask = pred_mask.reshape(B, T, W)
        actions = actions.reshape(B, T, -1)
        rollout, rollout_mask = [], []
        current, current_mask = token_states[:, 0], token_mask[:, 0]
        for step in range(T):
            current, current_mask = self.token_pred(
                current, current_mask, operations[:, step],
                positions[:, step], content[:, step], prompt_emb,
            )
            rollout.append(current)
            rollout_mask.append(current_mask)
        return (
            pred, pred_mask, torch.stack(rollout, 1),
            torch.stack(rollout_mask, 1), actions,
        )

    @torch.no_grad()
    def update_teachers(self, momentum: float) -> None:
        self.chunk_teacher.update(self.chunk_encoder, momentum)
        self.buffer_teacher.update(self.buffer_encoder, momentum)

    def forward(self, batch: dict) -> JEPAOutputs:
        if self.token_aligned:
            return self._forward_token_aligned(batch)
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
            if "alt_changed_tokens" in batch:
                with torch.no_grad():
                    Bc, Tc, Kc, Lc = batch["alt_changed_tokens"].shape
                    local_targets = self.chunk_anchor(
                        batch["alt_changed_tokens"].reshape(Bc * Tc * Kc, Lc)
                    ).reshape(Bc, Tc, Kc, -1)
                out.extras["cf_slot_pred"] = self.core.chunk_head(
                    out.extras["alt_preds"]
                )
                out.extras["cf_slot_tgt"] = local_targets
                out.extras["cf_slot_valid"] = (
                    batch["alt_changed_valid"] & out.step_mask.unsqueeze(-1)
                )
        if self._slot_tgt is not None:
            out.extras["slot_pred"] = self.core.chunk_head(out.preds)
            out.extras["slot_tgt"] = self._slot_tgt
        return out

    @staticmethod
    def select_adaptive_candidates(
        scores: torch.Tensor, valid: torch.Tensor, selected_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select predicted-high actions without consulting outcomes or goals."""
        if selected_k < 1:
            raise ValueError("selected_k must be positive for adaptive proposals")
        count = min(int(selected_k), scores.shape[-1])
        masked = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        indices = masked.topk(count, dim=-1).indices
        return indices, valid.gather(-1, indices)

    @staticmethod
    def select_positive_anchor_candidates(
        scores: torch.Tensor, exact_advantage: torch.Tensor,
        valid: torch.Tensor, selected_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Privileged positive anchor followed by predicted-hard negatives.

        This is a synthetic training control: exact terminal advantages choose
        only slot zero. Remaining candidates are the highest predicted GAR
        scores excluding that anchor. No deployment planner calls this path.
        """
        if selected_k < 1:
            raise ValueError("selected_k must be positive for adaptive proposals")
        count = min(int(selected_k), scores.shape[-1])
        minimum = torch.finfo(scores.dtype).min
        anchor = exact_advantage.to(scores.dtype).masked_fill(
            ~valid, minimum
        ).argmax(-1, keepdim=True)
        anchor_valid = valid.gather(-1, anchor)
        if count == 1:
            return anchor, anchor_valid
        hard_valid = valid.clone()
        hard_valid.scatter_(-1, anchor, False)
        hard = scores.masked_fill(~hard_valid, minimum).topk(
            count - 1, dim=-1
        ).indices
        return (
            torch.cat([anchor, hard], dim=-1),
            torch.cat([anchor_valid, hard_valid.gather(-1, hard)], dim=-1),
        )

    @staticmethod
    def proposal_ranking_scores(
        gar_scores: torch.Tensor, selection: str,
        example_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Hard GAR scores or deterministic target-independent random ranks."""
        if selection == "hard":
            return gar_scores
        if selection != "random":
            raise ValueError(f"unknown proposal_selection: {selection}")
        B, T, P = gar_scores.shape
        example = example_indices.to(
            device=gar_scores.device, dtype=torch.int64
        ).reshape(B, 1, 1)
        step = torch.arange(T, device=gar_scores.device, dtype=torch.int64)[
            None, :, None
        ]
        candidate = torch.arange(
            P, device=gar_scores.device, dtype=torch.int64
        )[None, None, :]
        hashed = (
            example * 73_856_093 + step * 19_349_663
            + candidate * 83_492_791 + 12_345
        ).remainder(2_147_483_647)
        return hashed.to(gar_scores.dtype) / 2_147_483_647

    def _adaptive_structured_candidates(
        self, out, batch: dict, prompt_emb: torch.Tensor,
        token_states: torch.Tensor, token_mask: torch.Tensor,
        pooled_states: torch.Tensor,
    ) -> None:
        """Mine broad GAR proposals, then execute only predicted-hard top K."""
        if self.selected_k < 1:
            raise ValueError(
                "proposal_pool_k data requires model.selected_k > 0"
            )
        B, T, P = batch["proposal_op"].shape
        # An insertion needs N+1 capacity even when the current example is the
        # longest sequence in its batch.  The extra masked slot does not alter
        # the pooled state or action context for non-insert operations.
        current = torch.nn.functional.pad(
            token_states[:, :-1], (0, 0, 0, 1)
        )
        current_mask = torch.nn.functional.pad(
            token_mask[:, :-1], (0, 1), value=False
        )
        broad_valid = batch["proposal_valid"] & out.step_mask.unsqueeze(-1)
        with torch.no_grad():
            broad_content = self.chunk_encoder.tok(
                batch["proposal_edit_content_token"]
            )
            broad_action = self.token_pred.encode_action(
                current.unsqueeze(2).expand(-1, -1, P, -1, -1).reshape(
                    B * T * P, current.shape[-2], current.shape[-1]
                ),
                current_mask.unsqueeze(2).expand(-1, -1, P, -1).reshape(
                    B * T * P, current_mask.shape[-1]
                ),
                batch["proposal_op"].reshape(-1),
                batch["proposal_edit_position"].reshape(-1),
                broad_content.reshape(B * T * P, -1),
            ).reshape(B, T, P, -1)
            broad_score = self.gar_head(torch.cat([
                pooled_states[:, :-1].unsqueeze(2).expand(-1, -1, P, -1),
                broad_action,
            ], dim=-1)).squeeze(-1)
            if self.proposal_selection == "positive_anchor":
                # Learned scores remain the auditable hard-negative ranking;
                # only selected slot zero receives privileged selection.
                ranking_score = broad_score
                if "gar_proposal_token_edit_target" not in batch:
                    raise ValueError(
                        "proposal_selection=positive_anchor requires "
                        "gar_teacher=token_edit_distance and exact proposal labels"
                    )
                exact_advantage = batch["gar_proposal_token_edit_target"]
                selected, selected_valid = self.select_positive_anchor_candidates(
                    broad_score, exact_advantage, broad_valid, self.selected_k
                )
            else:
                ranking_score = self.proposal_ranking_scores(
                    broad_score, self.proposal_selection, batch["index"]
                )
                selected, selected_valid = self.select_adaptive_candidates(
                    ranking_score, broad_valid, self.selected_k
                )
        K = selected.shape[-1]

        def gather(values: torch.Tensor) -> torch.Tensor:
            tail = values.shape[3:]
            index = selected.reshape(B, T, K, *([1] * len(tail))).expand(
                B, T, K, *tail
            )
            return values.gather(2, index)

        operations = gather(batch["proposal_op"])
        positions = gather(batch["proposal_edit_position"])
        content_ids = gather(batch["proposal_edit_content_token"])
        content = self.chunk_encoder.tok(content_ids)
        selected_current = current.unsqueeze(2).expand(
            -1, -1, K, -1, -1
        )
        selected_mask = current_mask.unsqueeze(2).expand(-1, -1, K, -1)
        selected_prompt = prompt_emb[:, None, None].expand(-1, T, K, -1)
        prediction, prediction_mask, actions = self.token_pred(
            selected_current.reshape(B * T * K, current.shape[-2], current.shape[-1]),
            selected_mask.reshape(B * T * K, current_mask.shape[-1]),
            operations.reshape(-1), positions.reshape(-1),
            content.reshape(B * T * K, -1),
            selected_prompt.reshape(B * T * K, -1),
            return_action=True,
        )
        prediction = prediction.reshape(B, T, K, prediction.shape[-2], -1)
        prediction_mask = prediction_mask.reshape(B, T, K, -1)
        actions = actions.reshape(B, T, K, -1)

        outcome_tokens = gather(batch["proposal_buffer_tokens"])
        outcome_mask = gather(batch["proposal_buffer_mask"])
        C, L = outcome_tokens.shape[-2:]
        with torch.no_grad():
            target, target_mask = self.encode_token_buffers_chunked(
                outcome_tokens.reshape(B, T * K, C, L), mode="teacher"
            )
        target = target.reshape(B, T, K, target.shape[-2], target.shape[-1])
        target_mask = target_mask.reshape(B, T, K, -1)
        width = max(prediction.shape[-2], target.shape[-2])
        if prediction.shape[-2] < width:
            prediction = torch.nn.functional.pad(
                prediction, (0, 0, 0, width - prediction.shape[-2])
            )
            prediction_mask = torch.nn.functional.pad(
                prediction_mask, (0, width - prediction_mask.shape[-1])
            )
        if target.shape[-2] < width:
            target = torch.nn.functional.pad(
                target, (0, 0, 0, width - target.shape[-2])
            )
            target_mask = torch.nn.functional.pad(
                target_mask, (0, width - target_mask.shape[-1])
            )
        out.extras.update({
            "adaptive_proposal_scores": broad_score,
            "adaptive_proposal_ranking_scores": ranking_score,
            "adaptive_proposal_selection": self.proposal_selection,
            "adaptive_selected_indices": selected,
            "adaptive_selected_valid": selected_valid,
            "cf_token_pred": prediction,
            "cf_token_pred_mask": prediction_mask,
            "cf_token_tgt": target.detach(),
            "cf_token_tgt_mask": target_mask,
            "cf_token_valid": selected_valid,
            "cf_structured_actions": actions,
        })
        if self.proposal_selection == "positive_anchor":
            anchor_advantage = batch["gar_proposal_token_edit_target"].gather(
                -1, selected[..., :1]
            ).squeeze(-1)
            valid_state = broad_valid.any(-1)
            positive_available = (
                (batch["gar_proposal_token_edit_target"] > 0) & broad_valid
            ).any(-1)
            anchor_positive = (
                anchor_advantage > 0
            ) & selected_valid[..., 0]
            denominator = valid_state.sum().clamp_min(1)
            out.extras.update({
                "adaptive_positive_anchor_candidate_terminal_privileged": True,
                "adaptive_positive_anchor_advantage": anchor_advantage.detach(),
                "adaptive_positive_anchor_available": positive_available,
                "adaptive_positive_anchor_is_positive": anchor_positive,
                "adaptive_positive_anchor_valid_state_count": valid_state.sum(),
                "adaptive_positive_anchor_available_count": (
                    positive_available & valid_state
                ).sum(),
                "adaptive_positive_anchor_coverage": (
                    (positive_available & valid_state).sum().float()
                    / denominator
                ),
                "adaptive_positive_anchor_least_bad_rate": (
                    ((~positive_available) & valid_state).sum().float()
                    / denominator
                ),
            })

    def _forward_token_aligned(self, batch: dict) -> JEPAOutputs:
        prompt_chunks = self.encode_chunks(batch["prompt_tokens"])
        prompt_weight = batch["prompt_mask"].unsqueeze(-1).to(prompt_chunks.dtype)
        prompt_emb = (prompt_chunks * prompt_weight).sum(1) / prompt_weight.sum(1).clamp_min(1)
        token_states, token_mask = self.encode_token_buffers(
            batch["buffer_tokens"], mode="online"
        )
        with torch.no_grad():
            token_targets, target_mask = self.encode_token_buffers(
                batch["buffer_tokens"], mode="teacher"
            )
            action_emb_tgt = self.encode_chunks(
                batch["action_tokens"], teacher=True
            )
        # Packing widths are determined by the same raw buffers and therefore
        # must agree between online and EMA encoders.
        batch["structured_token_mask"] = token_mask
        pred, pred_mask, rollout, rollout_mask, actions = (
            self._structured_transitions(batch, prompt_emb, token_states)
        )
        states = self._pool_tokens(token_states, token_mask)
        targets = self._pool_tokens(token_targets, target_mask)
        pooled_pred = self._pool_tokens(pred, pred_mask)
        out = self.core(
            states[:, 0], states[:, 1:], targets[:, 1:], actions,
            action_emb_tgt, batch["step_mask"],
            preds_override=pooled_pred,
        )
        out.rollout = self._pool_tokens(rollout, rollout_mask)
        out.extras.update({
            "token_predictions": pred,
            "token_prediction_mask": pred_mask,
            "token_rollout_predictions": rollout,
            "token_rollout_mask": rollout_mask,
            "token_targets": token_targets[:, 1:].detach(),
            "token_target_mask": target_mask[:, 1:],
        })
        if "proposal_op" in batch:
            self._adaptive_structured_candidates(
                out, batch, prompt_emb, token_states, token_mask, states
            )
        elif "alt_op" in batch:
            B, T, K = batch["alt_op"].shape
            current = token_states[:, :-1].unsqueeze(2).expand(-1, -1, K, -1, -1)
            current_mask = token_mask[:, :-1].unsqueeze(2).expand(-1, -1, K, -1)
            alt_content = self.chunk_encoder.tok(batch["alt_edit_content_token"])
            alt_prompt = prompt_emb[:, None, None].expand(-1, T, K, -1)
            cf_pred, cf_pred_mask, cf_actions = self.token_pred(
                current.reshape(B * T * K, current.shape[-2], current.shape[-1]),
                current_mask.reshape(B * T * K, current_mask.shape[-1]),
                batch["alt_op"].reshape(-1),
                batch["alt_edit_position"].reshape(-1),
                alt_content.reshape(B * T * K, -1),
                alt_prompt.reshape(B * T * K, -1),
                return_action=True,
            )
            C, L = batch["alt_buffer_tokens"].shape[-2:]
            with torch.no_grad():
                cf_target, cf_target_mask = self.encode_token_buffers_chunked(
                    batch["alt_buffer_tokens"].reshape(B, T * K, C, L),
                    mode="teacher",
                )
            cf_pred = cf_pred.reshape(B, T, K, cf_pred.shape[-2], cf_pred.shape[-1])
            cf_pred_mask = cf_pred_mask.reshape(B, T, K, -1)
            cf_actions = cf_actions.reshape(B, T, K, -1)
            cf_target = cf_target.reshape(B, T, K, cf_target.shape[-2], cf_target.shape[-1])
            cf_target_mask = cf_target_mask.reshape(B, T, K, -1)
            width = max(cf_pred.shape[-2], cf_target.shape[-2])
            if cf_pred.shape[-2] < width:
                cf_pred = torch.nn.functional.pad(
                    cf_pred, (0, 0, 0, width - cf_pred.shape[-2])
                )
                cf_pred_mask = torch.nn.functional.pad(
                    cf_pred_mask, (0, width - cf_pred_mask.shape[-1])
                )
            if cf_target.shape[-2] < width:
                cf_target = torch.nn.functional.pad(
                    cf_target, (0, 0, 0, width - cf_target.shape[-2])
                )
                cf_target_mask = torch.nn.functional.pad(
                    cf_target_mask, (0, width - cf_target_mask.shape[-1])
                )
            out.extras.update({
                "cf_token_pred": cf_pred,
                "cf_token_pred_mask": cf_pred_mask,
                "cf_token_tgt": cf_target.detach(),
                "cf_token_tgt_mask": cf_target_mask,
                "cf_token_valid": batch["alt_valid"] & out.step_mask.unsqueeze(-1),
                "cf_structured_actions": cf_actions,
            })
        # The clean terminal embedding is a privileged training target only.
        # The learned action-value head receives (state, action), not the goal.
        row = torch.arange(states.shape[0], device=states.device)
        goal_index = batch["step_mask"].sum(1).long().clamp(min=1)
        if "goal_buffer_tokens" in batch:
            # Replay traces end in a behavior-policy state, not necessarily the
            # clean solution.  Encode the separately labelled privileged goal
            # only for GAR targets; it is never an input to V(state, action).
            with torch.no_grad():
                goal_tokens, goal_mask = self.encode_token_buffers(
                    batch["goal_buffer_tokens"], mode="teacher"
                )
            goal = self._pool_tokens(goal_tokens, goal_mask)[:, 0]
            out.extras["gar_uses_separate_terminal_privileged_goal"] = True
        else:
            goal = targets[row, goal_index]
        target_prev = torch.cat([targets[:, :1], targets[:, 1:-1]], dim=1)
        future = []
        for step in range(out.step_mask.shape[1]):
            index = torch.minimum(
                torch.full_like(goal_index, step + self.gar_horizon), goal_index
            )
            future.append(targets[row, index])
        target_future = torch.stack(future, dim=1)
        ln = lambda value: torch.nn.functional.layer_norm(
            value, value.shape[-1:]
        )
        before = (ln(target_prev) - ln(goal).unsqueeze(1)).abs().mean(-1)
        after = (ln(target_future) - ln(goal).unsqueeze(1)).abs().mean(-1)
        out.extras["gar_action_value"] = self.gar_head(torch.cat([
            out.prev_states, out.actions
        ], dim=-1)).squeeze(-1)
        out.extras["gar_action_target"] = (before - after).detach()
        out.extras["gar_horizon"] = self.gar_horizon
        if "cf_structured_actions" in out.extras:
            # Alternatives are mechanically executed from exactly the same
            # current buffer.  The clean terminal EMA state is privileged
            # training supervision only; neither it nor the target advantage
            # enters the learned V(s,a) head.
            cf_actions = out.extras["cf_structured_actions"]
            current = states[:, :-1].unsqueeze(2).expand(
                -1, -1, cf_actions.shape[2], -1
            )
            cf_target = self._pool_tokens(
                out.extras["cf_token_tgt"], out.extras["cf_token_tgt_mask"]
            )
            cf_before = (ln(targets[:, :-1]) - ln(goal).unsqueeze(1)).abs().mean(-1)
            cf_after = (
                ln(cf_target) - ln(goal).unsqueeze(1).unsqueeze(2)
            ).abs().mean(-1)
            out.extras["gar_alt_action_value"] = self.gar_head(torch.cat([
                current, cf_actions
            ], dim=-1)).squeeze(-1)
            out.extras["gar_alt_action_target"] = (
                cf_before.unsqueeze(-1) - cf_after
            ).detach()
            out.extras["gar_alt_action_valid"] = out.extras["cf_token_valid"]
        if self.observed_action_decoder is not None:
            out.extras["observed_action_logits"] = self.observed_action_decoder(
                out.step_states - out.prev_states
            )
        return out
