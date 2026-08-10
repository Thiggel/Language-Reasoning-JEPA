"""Execution helpers for Stage 1 teacher passes and Stage 2 recurrence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from textjepa.models.action_transition import (
    ActionConditionedTransition,
    ResidualCapture,
    TransitionConfig,
    decoder_backbone,
    decoder_layers,
    install_upper_lora,
    load_trainable_state_dict,
)
from textjepa.objectives.predictive_state import (
    normalized_discount_weights,
    rollout_step_loss,
)


QWEN_SCREEN_LAYERS = {"target": 12, "sources": (18, 24)}
OLMO_MAIN_LAYERS = {"target": 8, "sources": (12, 16)}
TRANSFORMERS_VERSION = "5.13.0"


def require_transformers_runtime() -> None:
    import transformers
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"predictive-state experiments require Transformers "
            f"{TRANSFORMERS_VERSION}, got {transformers.__version__}"
        )


def architecture_defaults(model_id: str, layer_count: int) -> dict:
    lowered = model_id.lower()
    if "qwen2.5-0.5b" in lowered and layer_count == 24:
        return dict(QWEN_SCREEN_LAYERS)
    if "olmo-2" in lowered and layer_count == 16:
        return dict(OLMO_MAIN_LAYERS)
    if layer_count < 4:
        raise ValueError("decoder is too shallow for a cross-layer split")
    return {
        "target": max(1, layer_count // 2),
        "sources": (
            max(2, round(0.75 * layer_count)), layer_count
        ),
    }


def load_stage1_checkpoint(
    path,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[nn.Module, ActionConditionedTransition | None, dict]:
    """Recreate the pinned backbone, LoRA topology, and predictor strictly."""
    require_transformers_runtime()
    from transformers import AutoModelForCausalLM

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("kind") != "action_conditioned_cross_layer_stage1":
        raise ValueError("checkpoint is not a Stage 1 transition model")
    model = AutoModelForCausalLM.from_pretrained(
        payload["model_id"], revision=payload["model_revision"],
        dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    lora = payload.get("lora")
    if lora is not None:
        install_upper_lora(
            model,
            first_trainable_layer=int(lora["first_trainable_layer"]),
            rank=int(lora["rank"]), alpha=float(lora["alpha"]),
        )
    else:
        model.requires_grad_(False)
    predictor = None
    config_dict = payload.get("transition_config")
    if config_dict is not None:
        config_dict = dict(config_dict)
        config_dict["source_layers"] = tuple(config_dict["source_layers"])
        predictor = ActionConditionedTransition(
            TransitionConfig(**config_dict)
        ).to(device=device, dtype=dtype)
    modules = (model,) if predictor is None else (model, predictor)
    load_trainable_state_dict(payload["trainable_state"], *modules)
    return model, predictor, payload


def load_stage2_checkpoint(
    path,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[nn.Module, ActionConditionedTransition, dict]:
    """Recreate a recurrently trained checkpoint without the Stage 1 file."""
    require_transformers_runtime()
    from transformers import AutoModelForCausalLM

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("kind") != "action_conditioned_cross_layer_stage2":
        raise ValueError("checkpoint is not a Stage 2 transition model")
    stage1 = payload["stage1_payload"]
    model = AutoModelForCausalLM.from_pretrained(
        stage1["model_id"], revision=stage1["model_revision"],
        dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    lora = stage1.get("lora")
    if lora is not None:
        install_upper_lora(
            model,
            first_trainable_layer=int(lora["first_trainable_layer"]),
            rank=int(lora["rank"]), alpha=float(lora["alpha"]),
        )
    else:
        model.requires_grad_(False)
    config_dict = dict(stage1["transition_config"])
    config_dict["source_layers"] = tuple(config_dict["source_layers"])
    predictor = ActionConditionedTransition(
        TransitionConfig(**config_dict)
    ).to(device=device, dtype=dtype)
    load_trainable_state_dict(payload["trainable_state"], model, predictor)
    return model, predictor, payload


@dataclass
class TeacherOutput:
    logits: torch.Tensor
    states: dict[int, torch.Tensor]
    past_key_values: Any | None


def teacher_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    capture: ResidualCapture,
    attention_mask: torch.Tensor | None = None,
    use_cache: bool = False,
) -> TeacherOutput:
    capture.clear()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=use_cache,
        return_dict=True,
    )
    if not capture.values:
        raise RuntimeError("decoder hooks did not capture residual states")
    return TeacherOutput(
        logits=output.logits,
        states=dict(capture.values),
        past_key_values=getattr(output, "past_key_values", None),
    )


def transition_prediction(
    predictor: ActionConditionedTransition,
    model: nn.Module,
    states: Mapping[int, torch.Tensor],
    action_ids: torch.Tensor,
) -> torch.Tensor:
    action = None
    if predictor.config.uses_action:
        action = model.get_input_embeddings()(action_ids).detach()
    return predictor(states, action)


def dense_stage1_prediction(
    predictor: ActionConditionedTransition,
    model: nn.Module,
    teacher: TeacherOutput,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sources = {
        layer: teacher.states[layer][:, :-1]
        for layer in predictor.config.used_source_layers
    }
    prediction = transition_prediction(
        predictor, model, sources, input_ids[:, 1:]
    )
    target = teacher.states[predictor.config.target_layer][:, 1:].detach()
    return prediction, target


class UpperStackRunner:
    """Advance one injected residual through the original upper decoder.

    The runner targets decoder families such as Qwen2 and OLMo2 that expose
    `layers`, `rotary_emb`, `norm`, and a transformers Cache. Decode calls use
    one query token and explicit positions; an attention mask is unnecessary
    because no query has a future key in a one-token step.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        target_layer: int,
        source_layers: tuple[int, ...],
    ):
        self.model = model
        self.backbone = decoder_backbone(model)
        self.layers = decoder_layers(model)
        self.target_layer = int(target_layer)
        self.source_layers = tuple(map(int, source_layers))
        if not 1 <= target_layer < len(self.layers):
            raise ValueError("target layer must leave a non-empty upper stack")
        if any(layer <= target_layer or layer > len(self.layers)
               for layer in source_layers):
            raise ValueError("sources must be in the injected upper stack")
        for name in ("rotary_emb", "norm"):
            if not hasattr(self.backbone, name):
                raise TypeError(f"decoder backbone lacks {name}")
        if not hasattr(model, "lm_head"):
            raise TypeError("causal LM lacks lm_head")

    def step(
        self,
        injected: torch.Tensor,
        *,
        past_key_values: Any,
        position_index: int,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        if injected.ndim == 2:
            injected = injected[:, None, :]
        if injected.ndim != 3 or injected.shape[1] != 1:
            raise ValueError("upper-stack step expects B x 1 x D")
        position_ids = torch.full(
            (1, 1), int(position_index), dtype=torch.long,
            device=injected.device,
        )
        position_embeddings = self.backbone.rotary_emb(
            injected, position_ids
        )
        hidden = injected
        states: dict[int, torch.Tensor] = {}
        layer_types = getattr(self.backbone.config, "layer_types", None)
        for zero_index in range(self.target_layer, len(self.layers)):
            layer = self.layers[zero_index]
            kwargs = {
                "attention_mask": None,
                "position_ids": position_ids,
                "position_embeddings": position_embeddings,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            # Newer decoder APIs choose attention type inside the layer. No
            # mapping mask is needed for a single causal query.
            hidden = layer(hidden, **kwargs)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            one_index = zero_index + 1
            if one_index in self.source_layers:
                states[one_index] = hidden
        missing = set(self.source_layers) - set(states)
        if missing:
            raise RuntimeError(f"upper stack missed source layers: {missing}")
        normalized = self.backbone.norm(hidden)
        logits = self.model.lm_head(normalized)
        return states, logits


def crop_cache(past_key_values: Any, length: int) -> Any:
    if past_key_values is None:
        raise ValueError("recurrent rollout requires a KV cache")
    crop = getattr(past_key_values, "crop", None)
    if crop is None:
        raise TypeError("transformers cache does not support deterministic crop")
    crop(int(length))
    return past_key_values


@dataclass
class RolloutLossOutput:
    total: torch.Tensor
    state: torch.Tensor
    kl: torch.Tensor
    cross_entropy: torch.Tensor
    teacher_cross_entropy: torch.Tensor
    excess_nll: torch.Tensor
    top1_agreement: torch.Tensor
    top20_agreement: torch.Tensor
    log_rms_drift: torch.Tensor
    per_step: list[dict[str, float]]


def recurrent_rollout_loss(
    *,
    model: nn.Module,
    predictor: ActionConditionedTransition,
    input_ids: torch.Tensor,
    start_index: int,
    horizon: int,
    capture: ResidualCapture,
    discount: float = 0.97,
    ce_weight: float = 0.1,
) -> RolloutLossOutput:
    """Unroll teacher actions from one shared exact start position."""
    if start_index < 0 or start_index + horizon + 1 >= input_ids.shape[1]:
        raise ValueError("sequence is too short for requested rollout")
    teacher = teacher_forward(
        model, input_ids, capture=capture, attention_mask=None, use_cache=True
    )
    cache = crop_cache(teacher.past_key_values, start_index + 1)
    sources = {
        layer: teacher.states[layer][:, start_index:start_index + 1]
        for layer in predictor.config.used_source_layers
    }
    runner = UpperStackRunner(
        model,
        target_layer=predictor.config.target_layer,
        source_layers=predictor.config.source_layers,
    )
    weights = normalized_discount_weights(
        horizon, discount, device=input_ids.device
    )
    totals = {name: input_ids.new_zeros((), dtype=torch.float32) for name in (
        "total", "state", "kl", "cross_entropy", "teacher_cross_entropy",
        "excess_nll", "top1_agreement", "top20_agreement", "log_rms_drift"
    )}
    rows = []
    for step in range(1, horizon + 1):
        position = start_index + step
        action_ids = input_ids[:, position:position + 1]
        predicted = transition_prediction(
            predictor, model, sources, action_ids
        )
        sources, jump_logits = runner.step(
            predicted, past_key_values=cache, position_index=position
        )
        target = teacher.states[predictor.config.target_layer][
            :, position:position + 1
        ].detach()
        teacher_logits = teacher.logits[:, position:position + 1].detach()
        next_token = input_ids[:, position + 1:position + 2]
        step_loss = rollout_step_loss(
            predicted_state=predicted,
            target_state=target,
            jump_logits=jump_logits,
            teacher_logits=teacher_logits,
            next_token=next_token,
            ce_weight=ce_weight,
        )
        weight = weights[step - 1]
        for name in totals:
            totals[name] = totals[name] + weight * getattr(step_loss, name)
        rows.append({
            "step": step,
            "state": float(step_loss.state.detach()),
            "kl": float(step_loss.kl.detach()),
            "cross_entropy": float(step_loss.cross_entropy.detach()),
            "teacher_cross_entropy": float(step_loss.teacher_cross_entropy.detach()),
            "excess_nll": float(step_loss.excess_nll.detach()),
            "top1_agreement": float(step_loss.top1_agreement.detach()),
            "top20_agreement": float(step_loss.top20_agreement.detach()),
            "log_rms_drift": float(step_loss.log_rms_drift.detach()),
        })
    return RolloutLossOutput(per_step=rows, **totals)


def _sample_top_p(
    logits: torch.Tensor, *, temperature: float, top_p: float
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_prob, sorted_index = probabilities.sort(descending=True, dim=-1)
    cumulative = sorted_prob.cumsum(-1)
    remove = cumulative - sorted_prob >= top_p
    sorted_prob = sorted_prob.masked_fill(remove, 0)
    sorted_prob = sorted_prob / sorted_prob.sum(-1, keepdim=True)
    selected = torch.multinomial(sorted_prob, 1)
    return sorted_index.gather(-1, selected).squeeze(-1)


@torch.no_grad()
def generate_jump_actions(
    *,
    model: nn.Module,
    predictor: ActionConditionedTransition,
    prompt_ids: torch.Tensor,
    action_count: int,
    capture: ResidualCapture,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> torch.Tensor:
    """Sample actions from the jump model for generated-prefix replay."""
    if action_count < 1 or prompt_ids.shape[1] < 1:
        raise ValueError("prompt and action count must be non-empty")
    exact = teacher_forward(
        model, prompt_ids, capture=capture, attention_mask=None, use_cache=True
    )
    cache = exact.past_key_values
    position = prompt_ids.shape[1] - 1
    sources = {
        layer: exact.states[layer][:, -1:]
        for layer in predictor.config.used_source_layers
    }
    runner = UpperStackRunner(
        model, target_layer=predictor.config.target_layer,
        source_layers=predictor.config.source_layers,
    )
    current_logits = exact.logits[:, -1]
    actions = []
    for offset in range(1, action_count + 1):
        action = _sample_top_p(
            current_logits, temperature=temperature, top_p=top_p
        )
        actions.append(action)
        predicted = transition_prediction(
            predictor, model, sources, action[:, None]
        )
        sources, logits = runner.step(
            predicted, past_key_values=cache, position_index=position + offset
        )
        current_logits = logits[:, -1]
    return torch.stack(actions, dim=1)
