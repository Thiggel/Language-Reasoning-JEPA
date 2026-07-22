"""Shared accounting and checkpoint helpers for long scale runs."""

from __future__ import annotations

from pathlib import Path

import torch


def optimizer_steps(micro_batches_per_rank: int, accumulation: int) -> int:
    if accumulation < 1 or micro_batches_per_rank % accumulation:
        raise ValueError("micro-batches per rank must be divisible by accumulation")
    return micro_batches_per_rank // accumulation


def exposure_milestones(value) -> tuple[int, ...]:
    if value is None:
        return ()
    return tuple(sorted({int(item) for item in value if int(item) > 0}))


def crossed_milestones(previous: int, current: int, milestones) -> tuple[int, ...]:
    return tuple(mark for mark in milestones if previous < mark <= current)


def save_checkpoint(payload: dict, output: Path, exposure: int) -> Path:
    destination = output / f"checkpoint-examples-{int(exposure):09d}.pt"
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination
