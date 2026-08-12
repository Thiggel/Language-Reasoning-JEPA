"""Action-conditioned cross-layer transition models and lightweight LoRA."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

import torch
from torch import nn


TRANSITION_ARCHITECTURE = "action_conditioned_cross_layer_swiglu_v1"
LORA_TARGETS = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
})


def parameter_free_rms_norm(
    hidden: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    scale = hidden.float().square().mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (hidden.float() * scale).to(hidden.dtype)


def activation_rms(hidden: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return hidden.float().square().mean(dim=-1).add(eps).sqrt()


@dataclass(frozen=True)
class TransitionConfig:
    hidden_size: int
    source_layers: tuple[int, ...]
    target_layer: int
    action_dim: int | None = None
    projection_size: int | None = None
    action_projection_size: int | None = None
    predictor_width: int | None = None
    variant: str = "full"
    eps: float = 1e-6

    def __post_init__(self) -> None:
        variants = {
            "full", "no_action", "action_only", "same_layer", "nitp"
        }
        if self.variant not in variants:
            raise ValueError(f"unknown transition variant: {self.variant}")
        if self.hidden_size < 2 or self.target_layer < 1:
            raise ValueError("hidden size and layer indices must be positive")
        if not self.source_layers and self.variant != "action_only":
            raise ValueError("state-conditioned variants need a source layer")
        if self.variant == "action_only" and self.action_dim is None:
            raise ValueError("action-only prediction needs action embeddings")

    @property
    def d_projection(self) -> int:
        return self.projection_size or self.hidden_size // 2

    @property
    def d_action_projection(self) -> int:
        """Width of the action channel, independent of the state bottleneck.

        A single projection width would starve token identity along with the
        state: eight dimensions cannot separate a 151k-token vocabulary, so a
        tight bottleneck would degrade prediction for the uninteresting reason
        that the realized action became unreadable. Keeping this at full width
        bottlenecks only what the predictor may read about the state.
        """
        return self.action_projection_size or self.d_projection

    @property
    def d_predictor(self) -> int:
        return self.predictor_width or 2 * self.hidden_size

    @property
    def uses_action(self) -> bool:
        return self.variant in {"full", "action_only", "same_layer"}

    @property
    def has_action_channel(self) -> bool:
        # no_action retains an identically sized zero channel so its parameter
        # count matches full; only the realized-token information is removed.
        return self.variant in {
            "full", "no_action", "action_only", "same_layer"
        }

    @property
    def used_source_layers(self) -> tuple[int, ...]:
        if self.variant == "action_only":
            return ()
        if self.variant in {"nitp", "same_layer"}:
            return (self.source_layers[-1],)
        return self.source_layers

    @property
    def parameter_source_layers(self) -> tuple[int, ...]:
        # action_only retains zero-valued state channels so that a failure to
        # match the full predictor cannot be attributed to lower capacity.
        if self.variant in {"nitp", "same_layer"}:
            return (self.source_layers[-1],)
        return self.source_layers


class ActionConditionedTransition(nn.Module):
    """Bias-free projected-state/action SwiGLU transition.

    The module predicts the raw target residual. Direction and activation scale
    are supervised separately by the objective module.
    """

    def __init__(self, config: TransitionConfig):
        super().__init__()
        self.config = config
        projection = config.d_projection
        self.state_projections = nn.ModuleDict({
            str(layer): nn.Linear(config.hidden_size, projection, bias=False)
            for layer in config.parameter_source_layers
        })
        self.action_projection = None
        if config.has_action_channel:
            self.action_projection = nn.Linear(
                config.action_dim or config.hidden_size,
                config.d_action_projection, bias=False,
            )
        if not config.parameter_source_layers and not config.has_action_channel:
            raise ValueError("transition has no inputs")
        input_size = len(config.parameter_source_layers) * projection
        if config.has_action_channel:
            input_size += config.d_action_projection
        self.gate = nn.Linear(input_size, config.d_predictor, bias=False)
        self.value = nn.Linear(input_size, config.d_predictor, bias=False)
        self.skip = nn.Linear(input_size, config.hidden_size, bias=False)
        self.output = nn.Linear(
            config.d_predictor, config.hidden_size, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)

    def forward(
        self,
        sources: Mapping[int, torch.Tensor],
        action_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        pieces = []
        output_dtype = None
        for layer in self.config.parameter_source_layers:
            if self.config.variant == "action_only":
                if action_embedding is None:
                    raise ValueError("action embedding is required")
                normalized = torch.zeros(
                    *action_embedding.shape[:-1], self.config.hidden_size,
                    dtype=action_embedding.dtype, device=action_embedding.device,
                )
            else:
                if layer not in sources:
                    raise KeyError(f"missing source residual from layer {layer}")
                normalized = parameter_free_rms_norm(
                    sources[layer], self.config.eps
                )
            output_dtype = output_dtype or normalized.dtype
            projection = self.state_projections[str(layer)]
            pieces.append(projection(normalized.to(projection.weight.dtype)))
        if self.config.has_action_channel:
            if self.config.uses_action:
                if action_embedding is None:
                    raise ValueError("action embedding is required")
                normalized_action = parameter_free_rms_norm(
                    action_embedding.detach(), self.config.eps
                )
            else:
                reference = next(iter(sources.values()))
                normalized_action = torch.zeros_like(reference)
            output_dtype = output_dtype or normalized_action.dtype
            pieces.append(self.action_projection(
                normalized_action.to(self.action_projection.weight.dtype)
            ))
        combined = torch.cat(pieces, dim=-1)
        update = torch.nn.functional.silu(self.gate(combined))
        update = update * self.value(combined)
        prediction = self.skip(combined) + self.output(update)
        return prediction.to(output_dtype)

    def metadata(self) -> dict:
        return {
            "architecture": TRANSITION_ARCHITECTURE,
            "config": asdict(self.config),
            "parameters": sum(p.numel() for p in self.parameters()),
        }


class FrozenSufficiencyProbe(nn.Module):
    """Identical recurrent probe for arbitrary history and action horizons.

    Checkpoints are compared with the same initialization, parameter count,
    state-history length, action horizon, optimizer, and examples. Q1 versus
    Q8 uses the same recurrent architecture while exposing different history
    lengths, avoiding a parameter-count advantage for the history probe.
    """

    def __init__(self, hidden_size: int, source_count: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.source_count = int(source_count)
        self.state_projection = nn.Linear(
            source_count * hidden_size, hidden_size, bias=False
        )
        self.action_projection = nn.Linear(
            hidden_size, hidden_size, bias=False
        )
        self.state_recurrence = nn.GRU(
            hidden_size, hidden_size, batch_first=True
        )
        self.action_recurrence = nn.GRU(
            hidden_size, hidden_size, batch_first=True
        )
        self.gate = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.value = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.output = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=1e-3)

    def forward(
        self,
        state_history: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        if state_history.ndim != 3 or action_sequence.ndim != 3:
            raise ValueError("probe inputs must be B x time x features")
        states = self.state_projection(
            parameter_free_rms_norm(state_history)
        )
        actions = self.action_projection(
            parameter_free_rms_norm(action_sequence.detach())
        )
        _, state_summary = self.state_recurrence(states)
        _, action_summary = self.action_recurrence(actions)
        combined = torch.cat([state_summary[-1], action_summary[-1]], dim=-1)
        update = torch.nn.functional.silu(self.gate(combined))
        update = update * self.value(combined)
        return self.output(update)


class LoRALinear(nn.Module):
    """A frozen linear map plus a trainable rank-constrained update."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(torch.empty(
            rank, base.in_features, device=base.weight.device,
            dtype=torch.float32,
        ))
        self.lora_b = nn.Parameter(torch.zeros(
            base.out_features, rank, device=base.weight.device,
            dtype=torch.float32,
        ))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden)
        low_rank = torch.nn.functional.linear(
            hidden.to(self.lora_a.dtype), self.lora_a
        )
        low_rank = torch.nn.functional.linear(low_rank, self.lora_b)
        return base_output + low_rank.to(base_output.dtype) * self.scaling


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidates = [
        getattr(model, "model", None),
        getattr(model, "transformer", None),
    ]
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers
        blocks = getattr(candidate, "blocks", None)
        if isinstance(blocks, nn.ModuleList):
            return blocks
    raise TypeError("model does not expose decoder layers")


def decoder_backbone(model: nn.Module) -> nn.Module:
    layers = decoder_layers(model)
    for candidate in (
        getattr(model, "model", None), getattr(model, "transformer", None)
    ):
        if candidate is not None and (
            getattr(candidate, "layers", None) is layers
            or getattr(candidate, "blocks", None) is layers
        ):
            return candidate
    raise TypeError("could not resolve decoder backbone")


def _replace_lora_targets(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    prefix: str,
) -> list[str]:
    replaced = []
    for name, child in list(module.named_children()):
        qualified = f"{prefix}.{name}" if prefix else name
        if name in LORA_TARGETS and isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            replaced.append(qualified)
        elif not isinstance(child, LoRALinear):
            replaced.extend(_replace_lora_targets(
                child, rank=rank, alpha=alpha, prefix=qualified
            ))
    return replaced


def install_upper_lora(
    model: nn.Module,
    *,
    first_trainable_layer: int,
    rank: int = 16,
    alpha: float = 32.0,
) -> dict:
    """Freeze a causal LM and insert LoRA in one-indexed upper layers."""
    model.requires_grad_(False)
    layers = decoder_layers(model)
    if not 1 <= first_trainable_layer <= len(layers):
        raise ValueError("first_trainable_layer is outside the decoder")
    replaced = []
    for zero_index in range(first_trainable_layer - 1, len(layers)):
        replaced.extend(_replace_lora_targets(
            layers[zero_index], rank=rank, alpha=alpha,
            prefix=f"layers.{zero_index}",
        ))
    if not replaced:
        raise ValueError("no supported attention/MLP projections were found")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "first_trainable_layer": first_trainable_layer,
        "rank": rank,
        "alpha": alpha,
        "replaced_modules": replaced,
        "trainable_parameters": trainable,
    }


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_a
            yield module.lora_b


def trainable_state_dict(*modules: nn.Module) -> dict[str, torch.Tensor]:
    result = {}
    for module_index, module in enumerate(modules):
        trainable_ids = {
            id(parameter) for parameter in module.parameters()
            if parameter.requires_grad
        }
        for name, parameter in module.named_parameters():
            if id(parameter) in trainable_ids:
                result[f"{module_index}:{name}"] = parameter.detach().cpu()
    return result


def load_trainable_state_dict(
    state: Mapping[str, torch.Tensor], *modules: nn.Module
) -> None:
    named = {
        f"{module_index}:{name}": parameter
        for module_index, module in enumerate(modules)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if set(state) != set(named):
        missing = sorted(set(named) - set(state))
        extra = sorted(set(state) - set(named))
        raise ValueError(f"trainable state mismatch: missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, value in state.items():
            parameter = named[name]
            if parameter.shape != value.shape:
                raise ValueError(f"shape mismatch for {name}")
            parameter.copy_(value.to(parameter.device, parameter.dtype))


class ResidualCapture(AbstractContextManager):
    """Capture raw outputs of selected one-indexed decoder blocks."""

    def __init__(self, model: nn.Module, layer_indices: Iterable[int]):
        layers = decoder_layers(model)
        indices = sorted(set(map(int, layer_indices)))
        if any(index < 1 or index > len(layers) for index in indices):
            raise ValueError("capture layer is outside the decoder")
        self.values: dict[int, torch.Tensor] = {}
        self.handles = []
        for index in indices:
            def hook(_module, _inputs, output, layer=index):
                hidden = output[0] if isinstance(output, tuple) else output
                self.values[layer] = hidden
            self.handles.append(layers[index - 1].register_forward_hook(hook))

    def clear(self) -> None:
        self.values.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
