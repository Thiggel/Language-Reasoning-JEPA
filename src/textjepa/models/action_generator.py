"""State-conditioned intent-phrase generator head.

Mirrors ``ObservedActionDecoder`` (see ``delta_decoder.py``) in style, but the
conditioning signal is the CURRENT pooled state instead of a latent
displacement, and decoding is autoregressive.  The head answers "which intent
phrase comes next here?" with free text, so a planner can propose actions
without any candidate catalogue.  It is an auxiliary read-out: the caller
always feeds it a detached state (see ``DiscourseJEPA._action_generator``).
"""

from __future__ import annotations

import torch
from torch import nn


class StateConditionedActionGenerator(nn.Module):
    """Small autoregressive decoder for intent-phrase tokens given a state.

    Teacher-forced training predicts token ``l`` from the state and tokens
    ``< l``; a learned BOS embedding starts the sequence and PAD (id 0 in the
    shared synthetic vocabulary) acts as the end marker, exactly as in the
    LDAD token targets.
    """

    def __init__(
        self,
        d_state: int,
        vocab_size: int,
        max_len: int,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.max_len = int(max_len)
        self.tok = nn.Embedding(vocab_size, d_state)
        self.pos = nn.Parameter(torch.empty(1, self.max_len, d_state))
        self.bos = nn.Parameter(torch.empty(1, 1, d_state))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.bos, std=0.02)
        self.condition = nn.Sequential(
            nn.LayerNorm(d_state), nn.Linear(d_state, d_state)
        )
        layer = nn.TransformerEncoderLayer(
            d_state, n_heads, d_state * 2, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer, n_layers, norm=nn.LayerNorm(d_state)
        )
        self.token_head = nn.Linear(d_state, vocab_size)

    def _logits(
        self, condition: torch.Tensor, inputs: torch.Tensor
    ) -> torch.Tensor:
        """[n, D] conditioning + [n, L, D] shifted inputs -> [n, L, V]."""
        L = inputs.shape[1]
        h = inputs + self.pos[:, :L] + condition.unsqueeze(1)
        mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=inputs.device),
            diagonal=1,
        )
        return self.token_head(self.decoder(h, mask=mask))

    def forward(
        self, state: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced logits: [..., D] state + [..., L] tokens -> [..., L, V].

        The returned logits are aligned with ``tokens`` (position ``l``
        predicts ``tokens[..., l]``), so the caller's cross-entropy target is
        the observed phrase itself.
        """
        shape = tokens.shape[:-1]
        flat_state = state.reshape(-1, state.shape[-1])
        flat_tokens = tokens.reshape(-1, tokens.shape[-1])[:, : self.max_len]
        n, L = flat_tokens.shape
        inputs = torch.cat(
            [self.bos.expand(n, 1, -1), self.tok(flat_tokens[:, :-1])], dim=1
        )
        logits = self._logits(self.condition(flat_state), inputs)
        return logits.reshape(*shape, L, logits.shape[-1])

    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        k: int = 8,
        top_p: float = 1.0,
        max_len: int | None = None,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> list[list[int]]:
        """Nucleus-sample ``k`` phrases from one state; deduplicated.

        Sequences stop at the first PAD token (the learned end marker) and are
        returned as token-id lists in first-sampled order.  ``generator``
        makes the draw reproducible independently of the ambient global RNG;
        callers that plan deterministically must pass one.
        """
        if k < 1:
            raise ValueError("k must be positive")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must lie in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        length = self.max_len if max_len is None else min(
            int(max_len), self.max_len
        )
        condition = self.condition(state.reshape(1, -1)).expand(k, -1)
        inputs = self.bos.expand(k, 1, -1)
        tokens = torch.zeros(
            k, 0, dtype=torch.long, device=condition.device
        )
        for _ in range(length):
            logits = self._logits(condition, inputs)[:, -1] / temperature
            probs = logits.softmax(-1)
            if top_p < 1.0:
                order = torch.argsort(probs, dim=-1, descending=True)
                sorted_probs = probs.gather(-1, order)
                cumulative = sorted_probs.cumsum(-1) - sorted_probs
                sorted_probs = sorted_probs.masked_fill(
                    cumulative >= top_p, 0.0
                )
                probs = torch.zeros_like(probs).scatter(
                    -1, order, sorted_probs
                )
            nxt = torch.multinomial(probs, 1, generator=generator)
            tokens = torch.cat([tokens, nxt], dim=1)
            inputs = torch.cat([inputs, self.tok(nxt)], dim=1)
        out: list[list[int]] = []
        for row in tokens.tolist():
            ids: list[int] = []
            for token in row:
                if token == 0:  # PAD terminates the phrase
                    break
                ids.append(int(token))
            if ids and ids not in out:
                out.append(ids)
        return out
