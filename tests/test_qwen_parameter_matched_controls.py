from types import SimpleNamespace

import torch
from torch import nn

from textjepa.models.qwen_parameter_matched import (
    ParameterMatchedLMHead,
    ResidualSwiGLUBudget,
    expose_exact_pretrained_budget,
    optimizer_groups,
)


def test_residual_swiglu_uses_exact_budget_and_starts_as_identity():
    layer = ResidualSwiGLUBudget(width=4, parameter_budget=26)
    assert sum(parameter.numel() for parameter in layer.parameters()) == 26
    hidden = torch.randn(2, 3, 4)
    torch.testing.assert_close(layer(hidden), hidden)


def test_parameter_matched_head_changes_only_through_residual_layer():
    adapter = ResidualSwiGLUBudget(width=4, parameter_budget=26)
    base = nn.Linear(4, 7, bias=False)
    head = ParameterMatchedLMHead(adapter, base)
    hidden = torch.randn(2, 4)
    torch.testing.assert_close(head(hidden), base(hidden))


class _ToyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
            for _ in range(6)
        ])


def test_unfrozen_control_exposes_exact_scalar_budget():
    model = _ToyQwen()
    masks, accounting = expose_exact_pretrained_budget(
        model, parameter_budget=73, candidate_layer_count=4
    )
    assert accounting["active_trainable_parameters"] == 73
    assert len(masks) == 1
    loss = sum(
        parameter.sum()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    loss.backward()
    partial = masks[0]
    assert torch.count_nonzero(partial.parameter.grad) == partial.active


def test_partial_mask_is_never_weight_decayed():
    model = _ToyQwen()
    masks, _ = expose_exact_pretrained_budget(
        model, parameter_budget=73, candidate_layer_count=4
    )
    groups = optimizer_groups(model, masks, weight_decay=0.1)
    partial_id = id(masks[0].parameter)
    matching = [
        group for group in groups
        if any(id(parameter) == partial_id for parameter in group["params"])
    ]
    assert len(matching) == 1
    assert matching[0]["weight_decay"] == 0.0
