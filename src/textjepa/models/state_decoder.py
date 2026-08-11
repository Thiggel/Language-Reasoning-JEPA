"""State -> sentence read-out for a FROZEN JEPA backbone.

``StateConditionedActionGenerator`` (see ``action_generator.py``) renders the
*intent* of the next action from a state.  This module renders the *outcome*
sentence that the state itself encodes — "so the number of X is 3 plus 4 = 7 ."
— plus a linear answer classifier on the final state.

It is deliberately a pure read-out: nothing here is ever part of the JEPA
objective, the module is trained on detached states of an eval-mode backbone
whose parameters have ``requires_grad=False``, so no gradient can shape the
representation.  Style (conditioning, PAD-as-end, nucleus/greedy sampling)
mirrors the action generator so the two heads stay comparable.
"""

from __future__ import annotations

import torch
from torch import nn


class FrozenStateSentenceDecoder(nn.Module):
    """Autoregressive sentence decoder + answer head on frozen states.

    Teacher-forced training predicts token ``l`` of the step sentence from the
    state and tokens ``< l``; a learned BOS embedding starts the sequence and
    PAD (id 0 in the shared synthetic vocabulary) acts as the end marker.
    """

    def __init__(
        self,
        d_state: int,
        vocab_size: int,
        max_len: int,
        n_layers: int = 2,
        n_heads: int = 4,
        n_answers: int = 0,
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
        self.answer_head = (
            nn.Sequential(nn.LayerNorm(d_state), nn.Linear(d_state, n_answers))
            if n_answers
            else None
        )

    # ------------------------------------------------------------------ #
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
        the observed sentence itself.
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

    def answer_logits(self, state: torch.Tensor) -> torch.Tensor:
        """[..., D] final state -> [..., n_answers] residue-class logits."""
        if self.answer_head is None:
            raise RuntimeError("decoder was built without an answer head")
        return self.answer_head(state)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        state: torch.Tensor,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_len: int | None = None,
        greedy: bool = True,
        generator: torch.Generator | None = None,
    ) -> list[list[int]]:
        """Decode one sentence per state row; PAD terminates each sentence.

        ``state`` is [n, D] (or [D]).  Returns ``n`` token-id lists, in the
        input order, with the trailing PAD removed.  Greedy by default because
        the read-out is evaluated for accuracy, not diversity; ``greedy=False``
        nucleus-samples with ``top_p``/``temperature`` like the action
        generator, and ``generator`` makes that draw reproducible.
        """
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must lie in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if state.dim() == 1:
            state = state.unsqueeze(0)
        length = self.max_len if max_len is None else min(
            int(max_len), self.max_len
        )
        n = state.shape[0]
        condition = self.condition(state)
        inputs = self.bos.expand(n, 1, -1)
        tokens = torch.zeros(n, 0, dtype=torch.long, device=state.device)
        for _ in range(length):
            logits = self._logits(condition, inputs)[:, -1] / temperature
            if greedy:
                nxt = logits.argmax(-1, keepdim=True)
            else:
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
            if bool((tokens == 0).any(1).all()):
                break
        out: list[list[int]] = []
        for row in tokens.tolist():
            ids: list[int] = []
            for token in row:
                if token == 0:  # PAD terminates the sentence
                    break
                ids.append(int(token))
            out.append(ids)
        return out
