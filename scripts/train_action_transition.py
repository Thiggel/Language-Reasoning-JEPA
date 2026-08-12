#!/usr/bin/env python3
"""Train one Stage 1 action-conditioned cross-layer prediction cell."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM

from textjepa.analysis.predictive_state import (
    geometry_summary,
    matched_action_geometry,
)
from textjepa.data.predictive_state import load_token_blocks, sha256_path
from textjepa.models.action_transition import (
    ActionConditionedTransition,
    ResidualCapture,
    TransitionConfig,
    decoder_layers,
    install_upper_lora,
    lora_parameters,
    parameter_free_rms_norm,
    trainable_state_dict,
)
from textjepa.objectives.predictive_state import stage1_loss
from textjepa.training.predictive_state import (
    architecture_defaults,
    dense_stage1_prediction,
    teacher_forward,
    transition_prediction,
    require_transformers_runtime,
)


QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B"
QWEN_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
OLMO_MODEL_ID = "allenai/OLMo-2-0425-1B"
OLMO_REVISION = "a1847dff35000b4271fa70afc5db10fd29fedbdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-blocks", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-id", default=QWEN_MODEL_ID)
    parser.add_argument("--model-revision", default=QWEN_REVISION)
    parser.add_argument(
        "--variant",
        choices=("full", "no_action", "action_only", "same_layer", "nitp", "ntp_only"),
        default="full",
    )
    parser.add_argument("--backbone-mode", choices=("frozen", "lora"), default="lora")
    parser.add_argument("--target-layer", type=int)
    parser.add_argument("--source-layer", type=int, action="append")
    # Narrowing the projection is an information bottleneck, not merely a
    # smaller predictor: every input, including the skip path, is routed
    # through it, so the transition cannot be solved inside the predictor
    # without the backbone making its states more predictable.
    parser.add_argument("--projection-size", type=int)
    parser.add_argument("--action-projection-size", type=int)
    parser.add_argument("--predictor-width", type=int)
    parser.add_argument("--linear-predictor", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--predictor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--prediction-weight", type=float, default=0.1)
    parser.add_argument("--scale-weight", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--predictor-weight-decay", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("TEXTJEPA_SOURCE_REVISION")


def cosine_schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def sample_batch(dataset, size: int, generator: torch.Generator,
                 device: str, sequence_length: int | None) -> dict:
    indices = torch.randint(len(dataset), (size,), generator=generator)
    batch = {
        key: torch.stack([dataset[int(index)][key] for index in indices]).to(device)
        for key in ("input_ids", "attention_mask", "target_mask", "position_ids")
    }
    if sequence_length is not None:
        if sequence_length < 2 or sequence_length > batch["input_ids"].shape[1]:
            raise ValueError("invalid sequence-length crop")
        batch["input_ids"] = batch["input_ids"][:, :sequence_length]
        batch["attention_mask"] = batch["attention_mask"][:, :sequence_length]
        batch["target_mask"] = batch["target_mask"][:, :sequence_length - 1]
        batch["position_ids"] = batch["position_ids"][:, :sequence_length]
    return batch


def make_predictor(model, variant: str, target: int,
                   sources: tuple[int, ...], *,
                   projection_size: int | None = None,
                   action_projection_size: int | None = None,
                   predictor_width: int | None = None,
                   linear_only: bool = False,
                   ) -> ActionConditionedTransition:
    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = int(text_config.hidden_size)
    if variant == "same_layer":
        target = sources[-1]
    config = TransitionConfig(
        hidden_size=hidden_size,
        source_layers=sources,
        target_layer=target,
        action_dim=hidden_size,
        projection_size=projection_size,
        action_projection_size=action_projection_size,
        predictor_width=predictor_width,
        linear_only=linear_only,
        variant=variant,
    )
    return ActionConditionedTransition(config)


@torch.no_grad()
def evaluate(model, predictor, dataset, capture, args, generator) -> dict:
    model.eval()
    if predictor is not None:
        predictor.eval()
    totals: dict[str, float] = {}
    states_for_geometry = []
    source_for_geometry = []
    actions_for_geometry = []
    future_for_geometry = []
    for _ in range(min(args.eval_batches, len(dataset))):
        batch = sample_batch(
            dataset, args.microbatch_size, generator, args.device,
            args.sequence_length,
        )
        teacher = teacher_forward(
            model, batch["input_ids"], capture=capture,
            attention_mask=None, position_ids=batch["position_ids"], use_cache=False,
        )
        prediction = target = None
        if predictor is not None:
            prediction, target = dense_stage1_prediction(
                predictor, model, teacher, batch["input_ids"]
            )
        losses = stage1_loss(
            logits=teacher.logits,
            input_ids=batch["input_ids"],
            target_mask=batch["target_mask"],
            prediction=prediction,
            target_state=target,
            prediction_weight=args.prediction_weight,
            scale_weight=args.scale_weight,
        )
        rows = {"predictor_removed_nll": float(losses.ntp)}
        if losses.transition is not None:
            rows.update({
                "transition_cosine_loss": float(losses.transition.cosine),
                "transition_scale_loss": float(losses.transition.scale),
                "transition_normalized_mse": float(losses.transition.normalized_mse),
                "transition_mean_cosine": float(losses.transition.mean_cosine),
                "predicted_rms": float(losses.transition.predicted_rms),
                "target_rms": float(losses.transition.target_rms),
            })
            if predictor.config.uses_action:
                permuted = transition_prediction(
                    predictor, model,
                    {layer: teacher.states[layer][:, :-1]
                     for layer in predictor.config.used_source_layers},
                    batch["input_ids"][:, 1:].roll(1, dims=1),
                )
                cos = torch.nn.functional.cosine_similarity(
                    parameter_free_rms_norm(permuted).float(),
                    parameter_free_rms_norm(target).float(), dim=-1,
                )
                rows["permuted_action_cosine_loss"] = float(
                    ((1.0 - cos) * batch["target_mask"]).sum()
                    / batch["target_mask"].sum().clamp_min(1)
                )
            states_for_geometry.append(target.cpu())
        if predictor is not None and predictor.config.used_source_layers:
            current = torch.cat([
                parameter_free_rms_norm(teacher.states[layer][:, :-1])
                for layer in predictor.config.used_source_layers
            ], dim=-1)
            source_for_geometry.append(current.cpu())
            actions_for_geometry.append(batch["input_ids"][:, 1:].cpu())
            future_for_geometry.append(
                teacher.states[predictor.config.target_layer][:, 1:].cpu()
            )
        for key, value in rows.items():
            totals[key] = totals.get(key, 0.0) + value
    batches = min(args.eval_batches, len(dataset))
    result = {key: value / batches for key, value in totals.items()}
    if states_for_geometry:
        result["target_geometry"] = geometry_summary(torch.cat(states_for_geometry))
    if source_for_geometry:
        result["matched_action_geometry"] = matched_action_geometry(
            torch.cat(source_for_geometry), torch.cat(future_for_geometry),
            torch.cat(actions_for_geometry),
        )
    model.train(args.backbone_mode == "lora")
    if predictor is not None:
        predictor.train()
    return result


def save_checkpoint(path: Path, *, model, predictor, args, step: int,
                    transition_config, lora_accounting, dataset_metadata,
                    evaluations) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    modules = (model,) if predictor is None else (model, predictor)
    torch.save({
        "schema_version": 1,
        "kind": "action_conditioned_cross_layer_stage1",
        "source_revision": git_revision(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "variant": args.variant,
        "backbone_mode": args.backbone_mode,
        "step": step,
        "transition_config": transition_config,
        "lora": lora_accounting,
        "training": vars(args) | {"token_blocks": str(args.token_blocks),
                                  "output": str(args.output)},
        "dataset_metadata": dataset_metadata,
        "dataset_sha256": sha256_path(args.token_blocks),
        "evaluations": evaluations,
        "trainable_state": trainable_state_dict(*modules),
    }, path)


def main() -> None:
    args = parse_args()
    require_transformers_runtime()
    args.output = args.output or (
        Path(os.environ["RUN_DIR"]) / "model" if "RUN_DIR" in os.environ else None
    )
    if args.output is None:
        raise ValueError("--output or RUN_DIR is required")
    if min(args.steps, args.microbatch_size, args.gradient_accumulation,
           args.eval_every, args.eval_batches) < 1:
        raise ValueError("training/evaluation counts must be positive")
    if args.backbone_mode == "frozen" and args.variant == "ntp_only":
        raise ValueError("frozen NTP-only has no trainable parameters")
    dtype = getattr(torch, args.dtype)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    train_data, dataset_metadata = load_token_blocks(args.token_blocks, "train")
    validation_data, validation_metadata = load_token_blocks(
        args.token_blocks, "validation"
    )
    if validation_metadata != dataset_metadata:
        raise AssertionError("train/validation metadata diverged")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.model_revision, dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    layer_count = len(decoder_layers(model))
    defaults = architecture_defaults(args.model_id, layer_count)
    target_layer = args.target_layer or defaults["target"]
    source_layers = tuple(args.source_layer or defaults["sources"])
    capture_target = source_layers[-1] if args.variant == "same_layer" else target_layer
    lora_accounting = None
    if args.backbone_mode == "lora":
        lora_accounting = install_upper_lora(
            model, first_trainable_layer=target_layer + 1,
            rank=args.lora_rank, alpha=args.lora_alpha,
        )
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    predictor = None
    if args.variant != "ntp_only":
        predictor = make_predictor(
            model, args.variant, target_layer, source_layers,
            projection_size=args.projection_size,
            action_projection_size=args.action_projection_size,
            predictor_width=args.predictor_width,
            linear_only=args.linear_predictor,
        ).to(device=args.device).train()
    capture_layers = set(source_layers) | {capture_target}
    capture = ResidualCapture(model, capture_layers)
    groups = []
    if predictor is not None:
        groups.append({
            "params": list(predictor.parameters()),
            "lr": args.predictor_learning_rate,
            "weight_decay": args.predictor_weight_decay,
        })
    lora = list(lora_parameters(model))
    if lora:
        groups.append({
            "params": lora, "lr": args.lora_learning_rate,
            "weight_decay": 0.0,
        })
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    warmup = max(1, round(args.steps * args.warmup_fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_schedule(step, args.steps, warmup)
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.dtype == "float16" and args.device.startswith("cuda")
    )
    train_generator = torch.Generator().manual_seed(args.seed + 101)
    history, evaluations = [], {}
    tokens_seen = 0
    best_transition = float("inf")
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        for _ in range(args.gradient_accumulation):
            batch = sample_batch(
                train_data, args.microbatch_size, train_generator,
                args.device, args.sequence_length,
            )
            with torch.autocast(
                device_type="cuda", dtype=dtype,
                enabled=args.device.startswith("cuda") and dtype != torch.float32,
            ):
                teacher = teacher_forward(
                    model, batch["input_ids"], capture=capture,
                    attention_mask=None, position_ids=batch["position_ids"],
                    use_cache=False,
                )
                prediction = target = None
                if predictor is not None:
                    prediction, target = dense_stage1_prediction(
                        predictor, model, teacher, batch["input_ids"]
                    )
                losses = stage1_loss(
                    logits=teacher.logits, input_ids=batch["input_ids"],
                    target_mask=batch["target_mask"], prediction=prediction,
                    target_state=target,
                    prediction_weight=args.prediction_weight,
                    scale_weight=args.scale_weight,
                )
            scaler.scale(losses.total / args.gradient_accumulation).backward()
            rows = {"total": float(losses.total.detach()),
                    "ntp": float(losses.ntp.detach())}
            if losses.transition is not None:
                rows.update({
                    "cosine": float(losses.transition.cosine.detach()),
                    "scale": float(losses.transition.scale.detach()),
                })
            for key, value in rows.items():
                sums[key] = sums.get(key, 0.0) + value
            tokens_seen += int(batch["target_mask"].sum())
        scaler.unscale_(optimizer)
        parameters = [parameter for group in groups for parameter in group["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        row = {
            "step": step,
            **{key: value / args.gradient_accumulation for key, value in sums.items()},
            "grad_norm": float(grad_norm),
            "tokens_seen": tokens_seen,
            "elapsed_seconds": perf_counter() - started,
            "learning_rates": scheduler.get_last_lr(),
        }
        if step == 1 or step % 25 == 0:
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(
                model, predictor, validation_data, capture, args,
                torch.Generator().manual_seed(args.seed + 202),
            )
            evaluations[str(step)] = result
            print(json.dumps({"step": step, "evaluation": result}), flush=True)
            transition_metric = result.get("transition_cosine_loss", result["predictor_removed_nll"])
            configuration = None if predictor is None else asdict(predictor.config)
            save_checkpoint(
                args.output / "last.pt", model=model, predictor=predictor,
                args=args, step=step, transition_config=configuration,
                lora_accounting=lora_accounting,
                dataset_metadata=dataset_metadata, evaluations=evaluations,
            )
            if transition_metric < best_transition:
                best_transition = transition_metric
                save_checkpoint(
                    args.output / "best.pt", model=model, predictor=predictor,
                    args=args, step=step, transition_config=configuration,
                    lora_accounting=lora_accounting,
                    dataset_metadata=dataset_metadata, evaluations=evaluations,
                )
    metrics = {
        "schema_version": 1,
        "status": "completed",
        "source_revision": git_revision(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "variant": args.variant,
        "backbone_mode": args.backbone_mode,
        "predictor": None if predictor is None else predictor.metadata(),
        "lora": lora_accounting,
        "training": {
            "steps": args.steps, "tokens_seen": tokens_seen,
            "wall_seconds": perf_counter() - started,
            "history": history,
        },
        "evaluations": evaluations,
        "scientific_validity": "diagnostic_only" if args.backbone_mode == "frozen" else "not_assessed",
        "oracle_information": False,
        "candidate_privileged_information": False,
        "cross_project_information": False,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    capture.__exit__(None, None, None)


if __name__ == "__main__":
    main()
