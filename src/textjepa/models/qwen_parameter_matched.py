"""Parameter-matched Qwen controls for the nested language JEPA.

The controls deliberately keep the pinned Qwen backbone and tokenizer.  The
``added_capacity`` control appends native Qwen decoder layers and a residual
SwiGLU head whose parameter count exactly equals the active token-JEPA
parameter budget.  The ``unfrozen`` control leaves model capacity unchanged
and exposes exactly that many pretrained scalar weights to optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


TOKEN_JEPA_PARAMETER_BUDGET = 77_329_408
ADDED_NATIVE_LAYER_TYPES = (
    "linear_attention",
    "linear_attention",
    "linear_attention",
)


class ResidualSwiGLUBudget(nn.Module):
    """A residual SwiGLU layer with an exact requested parameter count."""

    def __init__(self, width: int, parameter_budget: int):
        super().__init__()
        if width < 1 or parameter_budget < 3 * width:
            raise ValueError("SwiGLU parameter budget is too small")
        per_rank = 3 * width
        rank, remainder = divmod(parameter_budget, per_rank)
        if rank < 1 or remainder >= width:
            raise ValueError(
                "exact budget requires a remainder smaller than model width"
            )
        self.width = int(width)
        self.parameter_budget = int(parameter_budget)
        self.gate = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(width, rank, bias=False)
        self.down = nn.Linear(rank, width, bias=False)
        self.tail = nn.Parameter(torch.zeros(remainder))
        nn.init.zeros_(self.down.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        update = self.down(
            torch.nn.functional.silu(self.gate(hidden)) * self.up(hidden)
        )
        if self.tail.numel():
            update = update.clone()
            update[..., : self.tail.numel()] += self.tail
        return hidden + update


class ParameterMatchedLMHead(nn.Module):
    """Apply the added residual layer before the original tied LM head."""

    def __init__(self, adapter: ResidualSwiGLUBudget, base_head: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.base_head = base_head

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.base_head(self.adapter(hidden))


@dataclass
class TrainableMask:
    parameter: nn.Parameter
    mask: torch.Tensor
    name: str
    active: int


def append_parameter_matched_capacity(
    model: nn.Module,
    *,
    original_layer_count: int,
    parameter_budget: int = TOKEN_JEPA_PARAMETER_BUDGET,
) -> dict:
    """Freeze Qwen and make only the exact added capacity trainable."""
    model.requires_grad_(False)
    layers = model.model.layers
    if len(layers) <= original_layer_count:
        raise ValueError("model does not contain appended decoder layers")
    added = list(layers[original_layer_count:])
    native_parameters = sum(
        parameter.numel()
        for layer in added
        for parameter in layer.parameters()
    )
    residual_budget = parameter_budget - native_parameters
    if residual_budget <= 0:
        raise ValueError("native layers exceed requested parameter budget")
    text_config = getattr(model.config, "text_config", model.config)
    width = int(text_config.hidden_size)
    base_head = model.lm_head
    reference = next(base_head.parameters())
    adapter = ResidualSwiGLUBudget(width, residual_budget).to(
        device=reference.device, dtype=reference.dtype
    )
    model.lm_head = ParameterMatchedLMHead(adapter, base_head)
    for layer in added:
        layer.requires_grad_(True)
    adapter.requires_grad_(True)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable != parameter_budget:
        raise AssertionError(
            f"added capacity has {trainable} parameters, expected "
            f"{parameter_budget}"
        )
    return {
        "parameter_budget": parameter_budget,
        "native_added_parameters": native_parameters,
        "residual_added_parameters": residual_budget,
        "added_layer_types": list(ADDED_NATIVE_LAYER_TYPES),
        "original_layer_count": original_layer_count,
        "active_trainable_parameters": trainable,
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


def expose_exact_pretrained_budget(
    model: nn.Module,
    *,
    parameter_budget: int = TOKEN_JEPA_PARAMETER_BUDGET,
    candidate_layer_count: int = 4,
) -> tuple[list[TrainableMask], dict]:
    """Expose exactly ``parameter_budget`` scalars in the top Qwen blocks.

    Whole tensors are preferred.  At most one tensor receives a deterministic
    flat prefix mask.  The masked tensor must be placed in a zero-weight-decay
    optimizer group so inactive entries remain unchanged.
    """
    model.requires_grad_(False)
    layers = list(model.model.layers[-candidate_layer_count:])
    named: list[tuple[str, nn.Parameter]] = []
    offset = len(model.model.layers) - candidate_layer_count
    for local_index, layer in enumerate(reversed(layers)):
        layer_index = len(model.model.layers) - 1 - local_index
        for name, parameter in layer.named_parameters():
            named.append((f"model.layers.{layer_index}.{name}", parameter))
    available = sum(parameter.numel() for _, parameter in named)
    if available < parameter_budget:
        raise ValueError("candidate pretrained layers are below budget")
    masks: list[TrainableMask] = []
    remaining = int(parameter_budget)
    full_parameters = 0
    for name, parameter in named:
        if remaining <= 0:
            break
        parameter.requires_grad_(True)
        if parameter.numel() <= remaining:
            remaining -= parameter.numel()
            full_parameters += parameter.numel()
            continue
        mask = torch.zeros(
            parameter.numel(), dtype=torch.bool, device=parameter.device
        )
        mask[:remaining] = True
        mask = mask.reshape(parameter.shape)
        parameter.register_hook(
            lambda gradient, active=mask: gradient * active
        )
        masks.append(TrainableMask(parameter, mask, name, remaining))
        remaining = 0
    if remaining:
        raise AssertionError("failed to expose the requested parameter budget")
    requires_grad_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return masks, {
        "parameter_budget": parameter_budget,
        "active_trainable_parameters": parameter_budget,
        "requires_grad_tensor_parameters": requires_grad_count,
        "partially_masked_parameters": [
            {"name": item.name, "active": item.active}
            for item in masks
        ],
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "candidate_layer_count": candidate_layer_count,
    }


def optimizer_groups(
    model: nn.Module,
    masks: Iterable[TrainableMask],
    *,
    weight_decay: float,
) -> list[dict]:
    """Build AdamW groups without decaying inactive masked scalars."""
    masked_ids = {id(item.parameter) for item in masks}
    decay, no_decay, partial = [], [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in masked_ids:
            partial.append(parameter)
        elif parameter.ndim < 2:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if partial:
        groups.append({"params": partial, "weight_decay": 0.0})
    return groups


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only tensors belonging to modules exposed for training."""
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if id(parameter) in trainable_ids
    }
