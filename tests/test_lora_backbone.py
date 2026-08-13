import pytest
import torch
from torch import nn

from textjepa.models.lora_backbone import (
    LoRAConfig,
    LoRALinear,
    attach_lora,
    lora_parameters,
    trainable_parameter_count,
)


class TinyAttention(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.mlp = nn.Linear(width, width)

    def forward(self, value):
        return self.q_proj(value) + self.k_proj(value) + self.mlp(value)


class TinyBackbone(nn.Module):
    def __init__(self, layers: int = 2, width: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(TinyAttention(width) for _ in range(layers))

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


def test_lora_starts_as_an_exact_identity():
    """B is zero-initialized, so training starts from the pinned model."""
    torch.manual_seed(0)
    model = TinyBackbone()
    value = torch.randn(3, 8)
    before = model(value)
    attach_lora(model, LoRAConfig(rank=2))
    assert torch.allclose(before, model(value), atol=1e-6)


def test_only_targeted_projections_are_adapted_and_base_is_frozen():
    model = TinyBackbone()
    adapted = attach_lora(model, LoRAConfig(rank=2))
    assert all(name.endswith(("q_proj", "k_proj")) for name in adapted)
    assert len(adapted) == 4  # two layers x {q,k}
    assert isinstance(model.layers[0].q_proj, LoRALinear)
    assert isinstance(model.layers[0].mlp, nn.Linear)  # untargeted, untouched
    assert not model.layers[0].q_proj.base.weight.requires_grad
    assert not model.layers[0].mlp.weight.requires_grad


def test_only_adapter_parameters_carry_gradient():
    torch.manual_seed(0)
    model = TinyBackbone()
    attach_lora(model, LoRAConfig(rank=2))
    model(torch.randn(3, 8)).square().sum().backward()
    adapters = lora_parameters(model)
    assert adapters
    # lora_a receives gradient; lora_b starts at zero so its grad may vanish,
    # but every trainable parameter must be an adapter parameter.
    assert trainable_parameter_count(model) == sum(
        p.numel() for p in adapters
    )
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and parameter.grad.abs().sum() > 0:
            assert name.endswith(("lora_a", "lora_b"))


def test_gradient_reaches_the_backbone_through_the_hidden_states():
    """The whole point: a JEPA loss on hidden states must reach the LM."""
    torch.manual_seed(0)
    model = TinyBackbone()
    attach_lora(model, LoRAConfig(rank=2))
    head = nn.Linear(8, 4)
    hidden = model(torch.randn(3, 8))
    assert hidden.requires_grad
    head(hidden).square().sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in lora_parameters(model)
    )


def test_unmatched_targets_raise_rather_than_silently_freezing():
    with pytest.raises(ValueError, match="would train as if frozen"):
        attach_lora(TinyBackbone(), LoRAConfig(rank=2, targets=("absent",)))


def test_rank_larger_than_projection_is_rejected():
    with pytest.raises(ValueError, match="rank exceeds"):
        attach_lora(TinyBackbone(width=4), LoRAConfig(rank=8))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rank": 0}, "rank must be positive"),
        ({"alpha": 0.0}, "alpha must be positive"),
        ({"dropout": 1.0}, "dropout must lie"),
        ({"targets": ()}, "at least one target"),
    ],
)
def test_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LoRAConfig(**kwargs)
