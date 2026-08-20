"""Render an action VECTOR back to executable action TEXT, given the prompt.

Why this module exists
----------------------
Every catalogue-free proposer in this repo has died at DECODING, not at
proposing.  The 2026-08-12 diagnosis found the structural reason: an action
phrase names entities, the entity-name space is combinatorial, and the
existing one-shot decoder (``models/delta_decoder.ObservedActionDecoder``)
reads a phrase out of a latent DISPLACEMENT with no access to the problem
text.  It therefore has to *invent* names it has never seen in that
combination, and it emits some other problem's phrase instead (parse rate
.000).

The fix this module implements: condition the decoder on the PROBLEM CONTEXT
as well as the action vector.  Entity names are already present in the
prompt, so they can be COPIED (via cross-attention over the frozen encoder's
context hidden states) rather than invented; the action vector then only has
to say WHICH role/operation and WHICH slots.

Detachment (the whole point)
----------------------------
The decoder is a SEPARATE artifact with SEPARATE parameters, trained on
``no_grad`` states of a frozen backbone.  It shares NOTHING with the encoder
-- in particular it has its OWN token embedding table and its OWN output
matrix, deliberately untied from each other and from ``encoder.tok``.  That
is the trap the ``objective.intent_prior_lm.detach_state`` round uncovered:
detaching hidden states is not enough if the output head is weight-tied to
the encoder's input embedding, because the larger half of the gradient then
flows into the encoder through the shared table.  Here no tensor is shared,
so the decoding objective cannot shape the representation at all.

Nothing here is part of the JEPA objective and no config key is added.
"""

from __future__ import annotations

import torch
from torch import nn

Tensor = torch.Tensor


class ContextActionDecoder(nn.Module):
    """Autoregressive action-phrase decoder on (action vector, context).

    ``action`` is a [B, d_state] action code (the flat-JEPA action vector:
    the frozen encoder's hidden at the last token of the intent phrase,
    encoded in context).  ``context`` is [B, L, d_state] frozen encoder
    hidden states over the prompt-plus-history tokens.  Cross-attention over
    ``context`` is what lets names be copied instead of invented.

    ``use_action`` / ``use_context`` exist so the two ablation controls
    (context-only, action-only) are the SAME architecture with one input
    zeroed -- not a different model with a different capacity.
    """

    def __init__(
        self,
        d_state: int,
        vocab_size: int,
        max_len: int = 24,
        d_model: int = 384,
        n_layers: int = 3,
        n_heads: int = 6,
        dropout: float = 0.0,
        use_action: bool = True,
        use_context: bool = True,
    ):
        super().__init__()
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.use_action = bool(use_action)
        self.use_context = bool(use_context)
        # Own embedding table; NOT tied to the encoder and NOT tied to the
        # output head (see the module docstring).
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.empty(1, self.max_len, d_model))
        self.bos = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.bos, std=0.02)
        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(d_state), nn.Linear(d_state, d_model)
        )
        self.act_proj = nn.Sequential(
            nn.LayerNorm(d_state), nn.Linear(d_state, d_model)
        )
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, n_layers, norm=nn.LayerNorm(d_model)
        )
        self.head = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------------------ #
    def _memory(self, action: Tensor, context: Tensor,
                context_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Cross-attention memory: [action slot] ++ context tokens.

        Returns (memory [B, 1+L, d_model], key_padding_mask [B, 1+L]).  The
        action always occupies slot 0 so the memory is never empty, even in
        the context-only / action-only controls.
        """
        B = action.shape[0]
        a = self.act_proj(action).unsqueeze(1)
        if not self.use_action:
            a = torch.zeros_like(a)
        c = self.ctx_proj(context)
        if not self.use_context:
            c = torch.zeros_like(c)
        mem = torch.cat([a, c], 1)
        pad = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=action.device),
             context_mask], 1,
        )
        if not self.use_context:
            pad = torch.cat(
                [pad[:, :1], torch.ones_like(context_mask)], 1)
        return mem, pad

    def _logits(self, action: Tensor, context: Tensor, context_mask: Tensor,
                inputs: Tensor) -> Tensor:
        T = inputs.shape[1]
        # The action code conditions EVERY decoder position, not only the
        # cross-attention memory slot: as one of ~400 memory tokens it would
        # be drowned out by the context and the ablation would be vacuous.
        cond = self.act_proj(action).unsqueeze(1)
        if not self.use_action:
            cond = torch.zeros_like(cond)
        h = inputs + self.pos[:, :T] + cond
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=inputs.device),
            diagonal=1,
        )
        mem, mem_pad = self._memory(action, context, context_mask)
        out = self.decoder(
            h, mem, tgt_mask=mask, memory_key_padding_mask=mem_pad
        )
        return self.head(out)

    def forward(self, action: Tensor, context: Tensor, context_mask: Tensor,
                tokens: Tensor) -> Tensor:
        """Teacher-forced logits aligned with ``tokens``: [B, T] -> [B, T, V].

        ``context_mask`` is True on PAD positions (torch key-padding
        convention).
        """
        emb = self.tok(tokens[:, :-1])
        inputs = torch.cat([self.bos.expand(tokens.shape[0], -1, -1), emb], 1)
        return self._logits(action, context, context_mask, inputs)

    @torch.no_grad()
    def generate(self, action: Tensor, context: Tensor, context_mask: Tensor,
                 eos_id: int = 0, max_len: int | None = None
                 ) -> list[list[int]]:
        """Greedy decode; ``eos_id`` (PAD) terminates and is not emitted."""
        cap = int(max_len or self.max_len)
        B = action.shape[0]
        inputs = self.bos.expand(B, -1, -1)
        done = torch.zeros(B, dtype=torch.bool, device=action.device)
        out: list[list[int]] = [[] for _ in range(B)]
        for _ in range(cap):
            logits = self._logits(action, context, context_mask, inputs)
            nxt = logits[:, -1].argmax(-1)
            for b in range(B):
                if not bool(done[b]):
                    if int(nxt[b]) == eos_id:
                        done[b] = True
                    else:
                        out[b].append(int(nxt[b]))
            if bool(done.all()):
                break
            inputs = torch.cat([inputs, self.tok(nxt).unsqueeze(1)], 1)
        return out

    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        torch.save({"cfg": {
            "d_state": self.ctx_proj[0].normalized_shape[0],
            "vocab_size": self.head.out_features,
            "max_len": self.max_len, "d_model": self.d_model,
            "n_layers": len(self.decoder.layers),
            "n_heads": self.decoder.layers[0].self_attn.num_heads,
            "use_action": self.use_action, "use_context": self.use_context,
        }, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "ContextActionDecoder":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**blob["cfg"])
        model.load_state_dict(blob["state"])
        return model
