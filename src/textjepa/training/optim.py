"""Optimizer and schedule construction."""

from __future__ import annotations

import math

import torch


def build_optimizer(model, lr: float, weight_decay: float, betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() < 2 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def cosine_warmup(step: int, total: int, warmup: int, floor: float = 0.05) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * min(t, 1.0)))


def ema_momentum(step: int, total: int, start: float, end: float) -> float:
    return start + (end - start) * min(step / max(total, 1), 1.0)
