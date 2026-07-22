"""Adversarial GPU equivalence audit for block-diagonal sequence packing."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch

from textjepa.models.layers import (
    attention_backend_summary,
    encoder_stack,
    packed_encoder_forward,
)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(left, right, dim=0))


def prefix_mask(lengths: list[int], width: int, device) -> torch.Tensor:
    return (
        torch.arange(width, device=device).unsqueeze(0)
        < torch.tensor(lengths, device=device).unsqueeze(1)
    )


def one_case(seed: int, lengths: list[int], dimension: int, layers: int) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    width = max(lengths)
    valid = prefix_mask(lengths, width, device)
    dense_model = encoder_stack(
        dimension, layers, 8, 4.175, 0.0, attention_backend="auto"
    ).to(device).train()
    packed_model = copy.deepcopy(dense_model).train()
    dense_input = torch.randn(
        len(lengths), width, dimension, device=device, requires_grad=True
    )
    packed_input = dense_input.detach().clone().requires_grad_(True)
    weight = torch.randn_like(dense_input) * valid.unsqueeze(-1)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dense = dense_model(dense_input, src_key_padding_mask=~valid)
        dense_loss = (dense * weight).sum()
    dense_loss.backward()
    dense_peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        packed = packed_encoder_forward(packed_model, packed_input, valid)
        packed_loss = (packed * weight).sum()
    packed_loss.backward()
    packed_peak = torch.cuda.max_memory_allocated() / 2**30

    forward_delta = (packed[valid] - dense[valid]).float()
    input_grad_delta = (
        packed_input.grad[valid] - dense_input.grad[valid]
    ).float()
    dense_parameter_grad = torch.cat([
        parameter.grad.float().reshape(-1)
        for parameter in dense_model.parameters()
    ])
    packed_parameter_grad = torch.cat([
        parameter.grad.float().reshape(-1)
        for parameter in packed_model.parameters()
    ])

    # Same layout, changed sequence zero: every other packed output must be
    # independent up to exact kernel determinism.
    changed = packed_input.detach().clone()
    changed[0, valid[0]] += 1000
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        changed_output = packed_encoder_forward(packed_model, changed, valid)
    isolation_delta = (
        changed_output[1:][valid[1:]] - packed.detach()[1:][valid[1:]]
    ).float()

    metrics = {
        "seed": seed,
        "lengths": lengths,
        "total_tokens": int(valid.sum()),
        "forward_cosine": cosine(packed[valid], dense[valid]),
        "forward_mean_abs": float(forward_delta.abs().mean()),
        "forward_max_abs": float(forward_delta.abs().max()),
        "input_grad_cosine": cosine(
            packed_input.grad[valid], dense_input.grad[valid]
        ),
        "input_grad_mean_abs": float(input_grad_delta.abs().mean()),
        "parameter_grad_cosine": cosine(
            packed_parameter_grad, dense_parameter_grad
        ),
        "cross_example_max_abs": float(isolation_delta.abs().max()),
        "dense_peak_gib": dense_peak,
        "packed_peak_gib": packed_peak,
        "backends": attention_backend_summary(packed_model),
    }
    assert metrics["forward_cosine"] > 0.999, metrics
    assert metrics["forward_mean_abs"] < 0.02, metrics
    assert metrics["input_grad_cosine"] > 0.995, metrics
    assert metrics["input_grad_mean_abs"] < 0.03, metrics
    assert metrics["parameter_grad_cosine"] > 0.995, metrics
    assert metrics["cross_example_max_abs"] == 0.0, metrics
    assert all(torch.isfinite(parameter.grad).all()
               for parameter in packed_model.parameters()), metrics
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path(os.environ["RUN_DIR"]) if "RUN_DIR" in os.environ else None,
    )
    args = parser.parse_args()
    if args.out is None:
        parser.error("--out is required when RUN_DIR is not set")
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [
        one_case(1701, [129, 67, 11, 1], 832, 2),
        one_case(1702, [97, 96, 33], 832, 2),
        one_case(1703, [64, 31, 17, 8, 4, 2, 1], 832, 2),
    ]
    payload = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "cases": cases,
        "passed": True,
    }
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
