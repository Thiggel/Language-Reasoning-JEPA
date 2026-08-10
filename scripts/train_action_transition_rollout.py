#!/usr/bin/env python3
"""Train one Stage 2 recurrent-rollout horizon from a Stage 1 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch

from textjepa.data.predictive_state import load_token_blocks, sha256_path
from textjepa.models.action_transition import (
    ResidualCapture,
    lora_parameters,
    trainable_state_dict,
)
from textjepa.training.predictive_state import (
    generate_jump_actions,
    load_stage1_checkpoint,
    recurrent_rollout_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--token-blocks", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--predictor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--discount", type=float, default=0.97)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--on-policy-fraction", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))


def sample_valid_batch(dataset, *, size: int, sequence_length: int | None,
                       horizon: int, generator: torch.Generator,
                       device: str) -> tuple[torch.Tensor, int]:
    for _ in range(128):
        indices = torch.randint(len(dataset), (size,), generator=generator)
        inputs = torch.stack([
            dataset[int(index)]["input_ids"] for index in indices
        ])
        target_mask = torch.stack([
            dataset[int(index)]["target_mask"] for index in indices
        ])
        if sequence_length is not None:
            inputs = inputs[:, :sequence_length]
            target_mask = target_mask[:, :sequence_length - 1]
        maximum_start = inputs.shape[1] - horizon - 2
        if maximum_start < 0:
            raise ValueError("sequence is shorter than horizon plus two")
        start = int(torch.randint(
            maximum_start + 1, (), generator=generator
        ))
        if bool(target_mask[:, start:start + horizon + 1].all()):
            return inputs.to(device), start
    raise RuntimeError("could not sample a rollout that stays within documents")


def save(path: Path, *, model, predictor, payload, args, step, history) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1,
        "kind": "action_conditioned_cross_layer_stage2",
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "stage1_sha256": sha256_path(args.stage1_checkpoint),
        "stage1_payload": {
            key: payload[key] for key in (
                "model_id", "model_revision", "variant", "transition_config", "lora"
            )
        },
        "step": step,
        "horizon": args.horizon,
        "training": {
            **vars(args),
            "stage1_checkpoint": str(args.stage1_checkpoint),
            "token_blocks": str(args.token_blocks),
            "output": str(args.output),
        },
        "history": history,
        "trainable_state": trainable_state_dict(model, predictor),
    }, path)


def main() -> None:
    args = parse_args()
    args.output = args.output or (
        Path(os.environ["RUN_DIR"]) / "model" if "RUN_DIR" in os.environ else None
    )
    if args.output is None:
        raise ValueError("--output or RUN_DIR is required")
    if args.horizon < 1 or args.steps < 1:
        raise ValueError("horizon and steps must be positive")
    if not 0.0 <= args.on_policy_fraction <= 1.0:
        raise ValueError("on-policy fraction must be in [0, 1]")
    if args.horizon < 16 and args.on_policy_fraction:
        raise ValueError("generated-prefix replay starts at horizon 16")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    model, predictor, stage1 = load_stage1_checkpoint(
        args.stage1_checkpoint, device=args.device, dtype=dtype
    )
    if predictor is None:
        raise ValueError("NTP-only checkpoint has no recurrent transition")
    dataset, metadata = load_token_blocks(args.token_blocks, "train")
    capture = ResidualCapture(
        model,
        set(predictor.config.source_layers) | {predictor.config.target_layer},
    )
    model.train()
    predictor.train()
    groups = [{
        "params": list(predictor.parameters()),
        "lr": args.predictor_learning_rate,
        "weight_decay": 0.1,
    }]
    lora = list(lora_parameters(model))
    if lora:
        groups.append({
            "params": lora, "lr": args.lora_learning_rate,
            "weight_decay": 0.0,
        })
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    warmup = max(1, round(args.steps * args.warmup_fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: schedule(step, args.steps, warmup)
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.dtype == "float16" and args.device.startswith("cuda")
    )
    generator = torch.Generator().manual_seed(args.seed + 303)
    history = []
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        sums = {key: 0.0 for key in (
            "total", "state", "kl", "cross_entropy", "teacher_cross_entropy",
            "excess_nll", "top1_agreement", "top20_agreement", "log_rms_drift"
        )}
        replay_batches = 0
        for _ in range(args.gradient_accumulation):
            inputs, start = sample_valid_batch(
                dataset, size=args.microbatch_size,
                sequence_length=args.sequence_length, horizon=args.horizon,
                generator=generator, device=args.device,
            )
            use_replay = torch.rand((), generator=generator).item() < args.on_policy_fraction
            if use_replay:
                prompt = inputs[:, :start + 1]
                actions = generate_jump_actions(
                    model=model, predictor=predictor, prompt_ids=prompt,
                    action_count=args.horizon + 1, capture=capture,
                    temperature=args.temperature, top_p=args.top_p,
                )
                inputs = torch.cat([prompt, actions], dim=1)
                replay_batches += 1
            with torch.autocast(
                device_type="cuda", dtype=dtype,
                enabled=args.device.startswith("cuda") and dtype != torch.float32,
            ):
                losses = recurrent_rollout_loss(
                    model=model, predictor=predictor, input_ids=inputs,
                    start_index=start, horizon=args.horizon, capture=capture,
                    discount=args.discount, ce_weight=args.ce_weight,
                )
            scaler.scale(losses.total / args.gradient_accumulation).backward()
            for key in sums:
                sums[key] += float(getattr(losses, key).detach())
        scaler.unscale_(optimizer)
        parameters = [parameter for group in groups for parameter in group["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        row = {
            "step": step,
            **{key: value / args.gradient_accumulation for key, value in sums.items()},
            "generated_replay_batches": replay_batches,
            "grad_norm": float(grad_norm),
            "learning_rates": scheduler.get_last_lr(),
            "elapsed_seconds": perf_counter() - started,
        }
        if step == 1 or step % 25 == 0:
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save(
                args.output / "last.pt", model=model, predictor=predictor,
                payload=stage1, args=args, step=step, history=history,
            )
    metrics = {
        "schema_version": 1,
        "status": "completed",
        "horizon": args.horizon,
        "on_policy_fraction": args.on_policy_fraction,
        "stage1_sha256": sha256_path(args.stage1_checkpoint),
        "dataset_metadata": metadata,
        "history": history,
        "wall_seconds": perf_counter() - started,
        "oracle_information": False,
        "teacher_actions": args.on_policy_fraction < 1.0,
        "generated_prefix_replay": args.on_policy_fraction > 0.0,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    capture.__exit__(None, None, None)


if __name__ == "__main__":
    main()
