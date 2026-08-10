#!/usr/bin/env python3
"""Evaluate Stage 2 teacher-action drift and decode-time operating cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from textjepa.data.predictive_state import load_token_blocks
from textjepa.models.action_transition import ResidualCapture
from textjepa.training.predictive_state import (
    UpperStackRunner,
    load_stage1_checkpoint,
    load_stage2_checkpoint,
    recurrent_rollout_loss,
    teacher_forward,
    transition_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-blocks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, action="append")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--benchmark-actions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    seen = seen or set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return tensor_bytes(vars(value), seen)
    return 0


def valid_example(dataset, row: int, horizon: int) -> tuple[torch.Tensor, int]:
    example = dataset[row % len(dataset)]
    target_mask = example["target_mask"]
    for start in range(len(target_mask) - horizon):
        if bool(target_mask[start:start + horizon + 1].all()):
            return example["input_ids"], start
    raise RuntimeError("validation block has no within-document rollout")


@torch.no_grad()
def benchmark(model, predictor, capture, tokens, actions: int, device: str) -> dict:
    prompt_length = min(128, tokens.shape[1] - actions - 1)
    if prompt_length < 2:
        return {"skipped": "sequence_too_short"}
    prompt = tokens[:, :prompt_length]
    continuation = tokens[:, prompt_length:prompt_length + actions]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    full = model(prompt, use_cache=True, return_dict=True)
    full_cache = full.past_key_values
    full_prefill_bytes = tensor_bytes(full_cache)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = perf_counter()
    for position in range(actions):
        model(
            input_ids=continuation[:, position:position + 1],
            past_key_values=full_cache, use_cache=True, return_dict=True,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    full_seconds = perf_counter() - started
    full_final_bytes = tensor_bytes(full_cache)

    exact = teacher_forward(
        model, prompt, capture=capture, attention_mask=None, use_cache=True
    )
    jump_cache = exact.past_key_values
    jump_prefill_bytes = tensor_bytes(jump_cache)
    sources = {
        layer: exact.states[layer][:, -1:]
        for layer in predictor.config.used_source_layers
    }
    runner = UpperStackRunner(
        model, target_layer=predictor.config.target_layer,
        source_layers=predictor.config.source_layers,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = perf_counter()
    for offset in range(actions):
        action = continuation[:, offset:offset + 1]
        predicted = transition_prediction(predictor, model, sources, action)
        sources, _ = runner.step(
            predicted, past_key_values=jump_cache,
            position_index=prompt_length + offset,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    jump_seconds = perf_counter() - started
    jump_final_bytes = tensor_bytes(jump_cache)
    return {
        "actions": actions,
        "full_seconds": full_seconds,
        "jump_seconds": jump_seconds,
        "throughput_ratio": full_seconds / max(jump_seconds, 1e-12),
        "full_tokens_per_second": actions / full_seconds,
        "jump_tokens_per_second": actions / jump_seconds,
        "full_prefill_cache_bytes": full_prefill_bytes,
        "jump_prefill_cache_bytes": jump_prefill_bytes,
        "full_decode_cache_growth_bytes": full_final_bytes - full_prefill_bytes,
        "jump_decode_cache_growth_bytes": jump_final_bytes - jump_prefill_bytes,
    }


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    header = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if header.get("kind") == "action_conditioned_cross_layer_stage2":
        model, predictor, payload = load_stage2_checkpoint(
            args.checkpoint, device=args.device, dtype=dtype
        )
    else:
        model, predictor, payload = load_stage1_checkpoint(
            args.checkpoint, device=args.device, dtype=dtype
        )
    if predictor is None:
        raise ValueError("checkpoint has no transition predictor")
    model.eval()
    predictor.eval()
    dataset, metadata = load_token_blocks(args.token_blocks, "validation")
    horizons = args.horizon or [1, 4, 8, 16, 32, 64, 128, 256]
    capture = ResidualCapture(
        model,
        set(predictor.config.source_layers) | {predictor.config.target_layer},
    )
    results = {}
    first_tokens = None
    for horizon in horizons:
        sums: dict[str, float] = {}
        count = 0
        rows = []
        for batch_index in range(min(args.batches, len(dataset))):
            try:
                tokens, start = valid_example(dataset, batch_index, horizon)
            except RuntimeError:
                continue
            tokens = tokens[None].repeat(args.microbatch_size, 1).to(args.device)
            first_tokens = tokens if first_tokens is None else first_tokens
            output = recurrent_rollout_loss(
                model=model, predictor=predictor, input_ids=tokens,
                start_index=start, horizon=horizon, capture=capture,
            )
            for key in (
                "state", "kl", "cross_entropy", "teacher_cross_entropy",
                "excess_nll", "top1_agreement", "top20_agreement",
                "log_rms_drift",
            ):
                sums[key] = sums.get(key, 0.0) + float(getattr(output, key))
            rows.append(output.per_step)
            count += 1
        if count:
            results[str(horizon)] = {
                "batches": count,
                **{key: value / count for key, value in sums.items()},
                "per_step_batches": rows,
            }
        else:
            results[str(horizon)] = {"batches": 0, "skipped": "no_valid_span"}
    timing = (
        benchmark(model, predictor, capture, first_tokens,
                  args.benchmark_actions, args.device)
        if first_tokens is not None else {"skipped": "no_valid_batch"}
    )
    report = {
        "schema_version": 1,
        "checkpoint_kind": payload["kind"],
        "checkpoint": str(args.checkpoint),
        "model_id": payload.get("model_id", payload.get("stage1_payload", {}).get("model_id")),
        "teacher_action_open_loop": results,
        "decode_benchmark": timing,
        "dataset_metadata": metadata,
        "scientific_validity": "not_assessed",
        "oracle_information": False,
        "candidate_privileged_information": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    capture.__exit__(None, None, None)


if __name__ == "__main__":
    main()
