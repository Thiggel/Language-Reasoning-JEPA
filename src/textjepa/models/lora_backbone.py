"""LoRA adapters for putting the reference LM into the JEPA gradient path.

The staged pipeline treats the LM as a fixed feature extractor: hidden states
are collected once and the hierarchy trains on cached tensors. That makes the
LM unable to adapt to what the planner needs from it -- in particular it keeps
emitting free-form prose when the latent space was built from canonical iGSM
sentences.

This module makes the backbone trainable at low rank so `E0/P0/E0_to_1/A1/P1`
and the LM can be optimized jointly. LoRA is implemented here rather than via
`peft` so the shared cluster environments stay unmodified.

Collapse warning: with a trainable backbone the cheapest way to satisfy a JEPA
prediction loss is to make representations uninformative. Anti-collapse
regularization and the effective-rank / symbolic-purity tripwires are not
optional in this mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn


DEFAULT_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = field(default=DEFAULT_TARGETS)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must lie in [0, 1)")
        if not self.targets:
            raise ValueError("at least one target module substring is required")


class LoRALinear(nn.Module):
    """Frozen base projection plus a trainable low-rank update.

    ``y = W0 x + (alpha / r) * B A x`` with ``B`` zero-initialized, so the
    wrapped module is an exact identity at step 0 and training starts from the
    pinned reference model's behaviour.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear wraps nn.Linear only")
        if config.rank > min(base.in_features, base.out_features):
            raise ValueError(
                "LoRA rank exceeds the wrapped projection's dimensions"
            )
        self.base = base
        self.base.requires_grad_(False)
        self.rank = config.rank
        self.scaling = config.alpha / config.rank
        self.lora_a = nn.Parameter(
            torch.empty(config.rank, base.in_features, dtype=base.weight.dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, config.rank, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.dropout(value) @ self.lora_a.T @ self.lora_b.T
        return self.base(value) + self.scaling * update


def attach_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Freeze ``model`` and wrap every matching ``nn.Linear`` with LoRA.

    Returns the qualified names that were adapted so callers can record them
    in run provenance. Raises if nothing matched, since a silently
    un-adapted backbone would look like a working end-to-end run while
    actually reproducing the frozen-LM setup.
    """

    model.requires_grad_(False)
    adapted: list[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            qualified = f"{name}.{child_name}" if name else child_name
            if not any(target in child_name for target in config.targets):
                continue
            setattr(module, child_name, LoRALinear(child, config))
            adapted.append(qualified)
    if not adapted:
        raise ValueError(
            f"no nn.Linear matched LoRA targets {config.targets}; the backbone "
            "would train as if frozen"
        )
    return adapted


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Trainable adapter parameters, in a deterministic order."""

    return [
        parameter
        for name, parameter in sorted(model.named_parameters())
        if parameter.requires_grad and (
            name.endswith("lora_a") or name.endswith("lora_b")
        )
    ]


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
