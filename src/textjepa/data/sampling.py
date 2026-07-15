"""Deterministic sampling utilities for procedurally generated datasets."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


class FreshEpochSampler(Sampler[int]):
    """Yield a disjoint deterministic index block on every epoch.

    The procedural datasets key generation by ``(seed, index)``.  Offsetting
    indices by ``epoch * len(dataset)`` therefore gives fresh problems without
    mutating worker-local dataset copies (and remains deterministic with
    persistent DataLoader workers).  ``set_epoch`` also seeds the within-epoch
    permutation, so resumed or repeated runs reproduce exactly the same data.
    """

    def __init__(self, data_source, seed: int = 0, shuffle: bool = True):
        self.data_source = data_source
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        n = len(self.data_source)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(n, generator=generator).tolist()
        else:
            order = range(n)
        offset = self.epoch * n
        return iter(offset + i for i in order)
