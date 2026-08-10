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
    parser.add_argument("--perturbation-scale", type=float, default=1e-3)
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


def valid_example(
    dataset, row: int, horizon: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    example = dataset[row % len(dataset)]
    target_mask = example["target_mask"]
    for start in range(len(target_mask) - horizon):
        if bool(target_mask[start:start + horizon + 1].all()):
            return example["input_ids"], example["position_ids"], start
    raise RuntimeError("validation block has no within-document rollout")


def longest_document_segment(example: dict) -> torch.Tensor:
    """Return the longest independently attended segment in a packed block."""
    tokens = example["input_ids"]
    boundary = (~example["target_mask"]).nonzero(as_tuple=False).flatten()
    begins = [0] + [int(index) + 1 for index in boundary]
    ends = [int(index) + 1 for index in boundary] + [len(tokens)]
    begin, end = max(zip(begins, ends), key=lambda pair: pair[1] - pair[0])
    return tokens[begin:end]


@torch.no_grad()
def full_greedy_decode(model, prompt: torch.Tensor, actions: int) -> torch.Tensor:
    output = model(prompt, use_cache=True, return_dict=True)
    cache = output.past_key_values
    logits = output.logits[:, -1]
    generated = []
    for _ in range(actions):
        token = logits.argmax(-1)
        generated.append(token)
        output = model(
            input_ids=token[:, None], past_key_values=cache,
            use_cache=True, return_dict=True,
        )
        logits = output.logits[:, -1]
    return torch.stack(generated, dim=1)


@torch.no_grad()
def jump_greedy_decode(
    model, predictor, capture, prompt: torch.Tensor, actions: int,
    *, refresh_interval: int | None = None,
) -> torch.Tensor:
    exact = teacher_forward(
        model, prompt, capture=capture, attention_mask=None, use_cache=True
    )
    cache = exact.past_key_values
    sources = {
        layer: exact.states[layer][:, -1:]
        for layer in predictor.config.used_source_layers
    }
    runner = UpperStackRunner(
        model, target_layer=predictor.config.target_layer,
        source_layers=predictor.config.source_layers,
    )
    logits = exact.logits[:, -1]
    generated = []
    full_tokens = prompt
    for offset in range(actions):
        token = logits.argmax(-1)
        generated.append(token)
        full_tokens = torch.cat([full_tokens, token[:, None]], dim=1)
        if (
            refresh_interval is not None
            and (offset + 1) % refresh_interval == 0
            and offset + 1 < actions
        ):
            exact = teacher_forward(
                model, full_tokens, capture=capture,
                attention_mask=None, use_cache=True,
            )
            cache = exact.past_key_values
            sources = {
                layer: exact.states[layer][:, -1:]
                for layer in predictor.config.used_source_layers
            }
            logits = exact.logits[:, -1]
            continue
        predicted = transition_prediction(
            predictor, model, sources, token[:, None]
        )
        sources, next_logits = runner.step(
            predicted, past_key_values=cache,
            position_index=full_tokens.shape[1] - 1,
        )
        logits = next_logits[:, -1]
    return torch.stack(generated, dim=1)


@torch.no_grad()
def speculative_greedy_decode(
    model, predictor, capture, prompt: torch.Tensor, actions: int,
    *, draft_length: int = 8,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Exact greedy decoding with jump drafts and batched full verification."""
    committed = prompt
    accepted = drafted = verifier_passes = 0
    while committed.shape[1] - prompt.shape[1] < actions:
        remaining = actions - (committed.shape[1] - prompt.shape[1])
        length = min(draft_length, remaining)
        draft = jump_greedy_decode(
            model, predictor, capture, committed, length
        )
        drafted += length
        verification = model(
            torch.cat([committed, draft], dim=1),
            use_cache=False, return_dict=True,
        ).logits
        verifier_passes += 1
        begin = committed.shape[1] - 1
        verifier_tokens = verification[:, begin:begin + length].argmax(-1)
        mismatch = (draft != verifier_tokens).any(0).nonzero(as_tuple=False)
        if len(mismatch):
            index = int(mismatch[0])
            if index:
                committed = torch.cat([committed, draft[:, :index]], dim=1)
                accepted += index
            committed = torch.cat([
                committed, verifier_tokens[:, index:index + 1]
            ], dim=1)
        else:
            committed = torch.cat([committed, draft], dim=1)
            accepted += length
    generated = committed[:, prompt.shape[1]:prompt.shape[1] + actions]
    return generated, {
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / max(drafted, 1),
        "verifier_passes": verifier_passes,
    }


@torch.no_grad()
def autonomous_evaluation(
    model, predictor, capture, prompt: torch.Tensor, actions: int
) -> dict:
    exact = full_greedy_decode(model, prompt, actions)
    result = {}
    for label, refresh in [
        ("jump_only", None), ("refresh_16", 16),
        ("refresh_32", 32), ("refresh_64", 64),
    ]:
        generated = jump_greedy_decode(
            model, predictor, capture, prompt, actions,
            refresh_interval=refresh,
        )
        result[label] = {
            "token_agreement_with_full_greedy": float((generated == exact).float().mean()),
            "sequence_agreement_with_full_greedy": bool((generated == exact).all()),
        }
    speculative, statistics = speculative_greedy_decode(
        model, predictor, capture, prompt, actions
    )
    result["speculative_exact_verification"] = {
        **statistics,
        "matches_full_greedy": bool((speculative == exact).all()),
    }
    return result


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
        perturbation_rows = []
        for batch_index in range(min(args.batches, len(dataset))):
            try:
                tokens, position_ids, start = valid_example(
                    dataset, batch_index, horizon
                )
            except RuntimeError:
                continue
            tokens = tokens[None].repeat(args.microbatch_size, 1).to(args.device)
            position_ids = position_ids[None].repeat(
                args.microbatch_size, 1
            ).to(args.device)
            first_tokens = tokens if first_tokens is None else first_tokens
            with torch.no_grad():
                output = recurrent_rollout_loss(
                    model=model, predictor=predictor, input_ids=tokens,
                    start_index=start, horizon=horizon, capture=capture,
                    position_ids=position_ids,
                )
                torch.manual_seed(args.seed + 1000 + batch_index)
                perturbed = recurrent_rollout_loss(
                    model=model, predictor=predictor, input_ids=tokens,
                    start_index=start, horizon=horizon, capture=capture,
                    position_ids=position_ids,
                    initial_source_noise=args.perturbation_scale,
                )
            for key in (
                "state", "kl", "cross_entropy", "teacher_cross_entropy",
                "excess_nll", "top1_agreement", "top20_agreement",
                "log_rms_drift",
            ):
                sums[key] = sums.get(key, 0.0) + float(getattr(output, key))
            rows.append(output.per_step)
            perturbation_rows.append([
                changed["state"] - base["state"]
                for base, changed in zip(output.per_step, perturbed.per_step)
            ])
            count += 1
        if count:
            results[str(horizon)] = {
                "batches": count,
                **{key: value / count for key, value in sums.items()},
                "per_step_batches": rows,
                "perturbation_scale": args.perturbation_scale,
                "perturbation_state_error_delta_by_step": [
                    sum(batch[step] for batch in perturbation_rows) / count
                    for step in range(horizon)
                ],
            }
        else:
            results[str(horizon)] = {"batches": 0, "skipped": "no_valid_span"}
    timing = (
        benchmark(model, predictor, capture, first_tokens,
                  args.benchmark_actions, args.device)
        if first_tokens is not None else {"skipped": "no_valid_batch"}
    )
    segment = longest_document_segment(dataset[0]).to(args.device)[None]
    autonomous_actions = min(
        args.benchmark_actions, max(0, segment.shape[1] - 2)
    )
    autonomous = (
        autonomous_evaluation(
            model, predictor, capture,
            segment[:, :segment.shape[1] - autonomous_actions],
            autonomous_actions,
        )
        if autonomous_actions else {"skipped": "document_segment_too_short"}
    )
    report = {
        "schema_version": 1,
        "checkpoint_kind": payload["kind"],
        "checkpoint": str(args.checkpoint),
        "model_id": payload.get("model_id", payload.get("stage1_payload", {}).get("model_id")),
        "teacher_action_open_loop": results,
        "decode_benchmark": timing,
        "autonomous_decoding": autonomous,
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
