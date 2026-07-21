"""A fixed-capacity text readout trained on frozen state representations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FrozenFeatureDecoder(nn.Module):
    """Autoregressive GRU readout; gradients never enter source features."""

    def __init__(
        self, feature_dim: int, vocab_size: int, pad_id: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.condition = nn.Linear(feature_dim, hidden_dim)
        self.tokens = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def logits(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        features = features.detach()
        start = torch.full_like(targets[:, :1], self.pad_id)
        inputs = torch.cat([start, targets[:, :-1]], dim=1)
        hidden = torch.tanh(self.condition(features)).unsqueeze(0)
        states, _ = self.gru(self.tokens(inputs), hidden)
        return self.output(states)

    def loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.logits(features, targets)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            ignore_index=self.pad_id,
        )

    @torch.no_grad()
    def generate(self, features: torch.Tensor, length: int) -> torch.Tensor:
        features = features.detach()
        hidden = torch.tanh(self.condition(features)).unsqueeze(0)
        token = torch.full(
            (len(features), 1), self.pad_id, dtype=torch.long,
            device=features.device,
        )
        output = []
        for _ in range(length):
            state, hidden = self.gru(self.tokens(token), hidden)
            token = self.output(state[:, -1:]).argmax(-1)
            output.append(token)
        return torch.cat(output, dim=1)
