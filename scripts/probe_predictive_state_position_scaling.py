#!/usr/bin/env python3
"""Transition error as a function of how much prefix the state must summarize.

The predictor sees a fixed-size state at position t: two residual vectors plus
the realized token. The target is what the frozen lower stack computes at t+1,
using attention over the whole prefix, which is O(t) memory. A fixed-size
summary of a growing prefix must eventually lose information, and that story
makes a sharp prediction: error should climb with position.

If instead error is flat in position, the residual error is not a compression
limit and more training or capacity may still reduce it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.data.predictive_state import load_token_blocks
from textjepa.models.action_transition import (
    ResidualCapture,
    parameter_free_rms_norm,
)
from textjepa.training.predictive_state import (
    dense_stage1_prediction,
    load_stage1_checkpoint,
    require_transformers_runtime,
    teacher_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", default=[],
                        metavar="NAME=PATH", required=True)
    parser.add_argument("--token-blocks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--buckets", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="bfloat16")
    return parser.parse_args()


@torch.no_grad()
def position_errors(model, predictor, capture, blocks, positions, args):
    """Mean cosine loss at each position, averaged over blocks."""
    totals = None
    counts = None
    for start in range(0, len(blocks), args.batch_size):
        tokens = blocks[start:start + args.batch_size].to(args.device)
        position_ids = positions[start:start + args.batch_size].to(args.device)
        teacher = teacher_forward(
            model, tokens, capture=capture, attention_mask=None,
            position_ids=position_ids, use_cache=False,
        )
        prediction, target = dense_stage1_prediction(
            predictor, model, teacher, tokens
        )
        cosine = torch.nn.functional.cosine_similarity(
            parameter_free_rms_norm(prediction).float(),
            parameter_free_rms_norm(target).float(), dim=-1,
        )
        loss = (1.0 - cosine).sum(0).cpu()
        if totals is None:
            totals = torch.zeros_like(loss)
            counts = torch.zeros_like(loss)
        totals += loss
        counts += tokens.shape[0]
    return totals / counts.clamp_min(1)


def main() -> None:
    args = parse_args()
    require_transformers_runtime()
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator().manual_seed(args.seed)
    dataset, metadata = load_token_blocks(args.token_blocks, "validation")
    chosen = torch.randperm(len(dataset), generator=generator)[:args.blocks]
    blocks = torch.stack([dataset[int(i)]["input_ids"] for i in chosen])
    positions = torch.stack([dataset[int(i)]["position_ids"] for i in chosen])

    report = {
        "schema_version": 1,
        "token_blocks": str(args.token_blocks),
        "blocks": int(args.blocks),
        "context_length": int(blocks.shape[1]),
        "buckets": args.buckets,
        "models": {},
        "oracle_information": False,
        "candidate_privileged_information": False,
        "cross_project_information": False,
    }

    for entry in args.checkpoint:
        name, _, path = entry.partition("=")
        model, predictor, payload = load_stage1_checkpoint(
            Path(path), device=args.device, dtype=dtype
        )
        model.eval()
        if predictor is None:
            raise ValueError(f"{name} has no predictor")
        predictor.eval()
        layers = set(predictor.config.used_source_layers) | {
            predictor.config.target_layer
        }
        with ResidualCapture(model, layers) as capture:
            errors = position_errors(
                model, predictor, capture, blocks, positions, args
            )
        edges = torch.linspace(0, len(errors), args.buckets + 1).long()
        bucketed = [
            {
                "first_position": int(edges[i]),
                "last_position": int(edges[i + 1]) - 1,
                "cosine_loss": float(errors[edges[i]:edges[i + 1]].mean()),
            }
            for i in range(args.buckets)
        ]
        report["models"][name] = {
            "checkpoint": str(path),
            "backbone_mode": payload.get("backbone_mode"),
            "overall_cosine_loss": float(errors.mean()),
            "first_bucket": bucketed[0]["cosine_loss"],
            "last_bucket": bucketed[-1]["cosine_loss"],
            "last_over_first": (
                bucketed[-1]["cosine_loss"] / max(bucketed[0]["cosine_loss"], 1e-9)
            ),
            "buckets": bucketed,
        }
        del model, predictor
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, values in report["models"].items():
        print(f"{name}: overall={values['overall_cosine_loss']:.4f} "
              f"first={values['first_bucket']:.4f} "
              f"last={values['last_bucket']:.4f} "
              f"ratio={values['last_over_first']:.2f}", flush=True)


if __name__ == "__main__":
    main()
