#!/usr/bin/env python3
"""Predictor-removed geometry, original-checkpoint CKA, and future retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from textjepa.analysis.predictive_state import geometry_summary, linear_cka
from textjepa.data.predictive_state import load_token_blocks
from textjepa.models.action_transition import ResidualCapture
from textjepa.training.predictive_state import (
    architecture_defaults,
    load_stage1_checkpoint,
    teacher_forward,
)


class RetrievalProbe(nn.Module):
    def __init__(self, width: int, projection: int = 128):
        super().__init__()
        self.query = nn.Linear(width, projection, bias=False)
        self.key = nn.Linear(width, projection, bias=False)

    def logits(self, query, key):
        query = torch.nn.functional.normalize(self.query(query), dim=-1)
        key = torch.nn.functional.normalize(self.key(key), dim=-1)
        return query @ key.T / 0.07


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-blocks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, action="append")
    parser.add_argument("--passage-tokens", type=int, default=16)
    parser.add_argument("--retrieval-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--geometry-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


@torch.no_grad()
def sample_pair(model, capture, dataset, *, layer, offset, passage,
                batch_size, generator, device):
    for _ in range(128):
        indices = torch.randint(len(dataset), (batch_size,), generator=generator)
        tokens = torch.stack([dataset[int(index)]["input_ids"] for index in indices])
        masks = torch.stack([dataset[int(index)]["target_mask"] for index in indices])
        maximum = tokens.shape[1] - offset - passage
        if maximum < 1:
            return None
        anchor = int(torch.randint(maximum, (), generator=generator))
        if not bool(masks[:, anchor:anchor + offset + passage].all()):
            continue
        output = teacher_forward(
            model, tokens.to(device), capture=capture,
            attention_mask=None, use_cache=False,
        )
        hidden = output.states[layer]
        query = hidden[:, anchor]
        key = hidden[:, anchor + offset:anchor + offset + passage].mean(1)
        return query.detach(), key.detach()
    raise RuntimeError("could not sample boundary-safe retrieval pairs")


def retrieval(model, capture, train_data, validation_data, *, layer, offset,
              width, args):
    generator = torch.Generator().manual_seed(args.seed + offset)
    probe = RetrievalProbe(width).to(args.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(args.retrieval_steps):
        pair = sample_pair(
            model, capture, train_data, layer=layer, offset=offset,
            passage=args.passage_tokens, batch_size=args.batch_size,
            generator=generator, device=args.device,
        )
        if pair is None:
            return {"skipped": "context_too_short"}
        query, key = pair
        logits = probe.logits(query.float(), key.float())
        labels = torch.arange(len(logits), device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    correct, total, losses = 0, 0, []
    probe.eval()
    with torch.no_grad():
        for _ in range(32):
            pair = sample_pair(
                model, capture, validation_data, layer=layer, offset=offset,
                passage=args.passage_tokens, batch_size=args.batch_size,
                generator=generator, device=args.device,
            )
            if pair is None:
                return {"skipped": "context_too_short"}
            query, key = pair
            logits = probe.logits(query.float(), key.float())
            labels = torch.arange(len(logits), device=logits.device)
            losses.append(float(torch.nn.functional.cross_entropy(logits, labels)))
            correct += int((logits.argmax(-1) == labels).sum())
            total += len(labels)
    return {
        "offset": offset,
        "negatives": args.batch_size - 1,
        "recall_at_1": correct / total,
        "cross_entropy": sum(losses) / len(losses),
        "examples": total,
    }


@torch.no_grad()
def collect_states(model, capture, dataset, *, layer, batches, batch_size, device):
    result = []
    for batch_index in range(min(batches, len(dataset))):
        begin = batch_index * batch_size
        rows = [(begin + offset) % len(dataset) for offset in range(batch_size)]
        tokens = torch.stack([dataset[row]["input_ids"] for row in rows]).to(device)
        output = teacher_forward(
            model, tokens, capture=capture, attention_mask=None, use_cache=False
        )
        result.append(output.states[layer].detach().cpu())
    return torch.cat(result)


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    model, predictor, payload = load_stage1_checkpoint(
        args.checkpoint, device=args.device, dtype=dtype
    )
    model.requires_grad_(False)
    model.eval()
    defaults = architecture_defaults(payload["model_id"], len(model.model.layers))
    final_layer = (
        predictor.config.source_layers[-1] if predictor is not None
        else defaults["sources"][-1]
    )
    train_data, metadata = load_token_blocks(args.token_blocks, "train")
    validation_data, _ = load_token_blocks(args.token_blocks, "validation")
    capture = ResidualCapture(model, (final_layer,))
    adapted = collect_states(
        model, capture, validation_data, layer=final_layer,
        batches=args.geometry_batches, batch_size=min(args.batch_size, 8),
        device=args.device,
    )
    retrieval_results = {}
    width = adapted.shape[-1]
    for offset in args.offset or [32, 128, 512]:
        retrieval_results[str(offset)] = retrieval(
            model, capture, train_data, validation_data, layer=final_layer,
            offset=offset, width=width, args=args,
        )
    capture.__exit__(None, None, None)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    original = AutoModelForCausalLM.from_pretrained(
        payload["model_id"], revision=payload["model_revision"],
        dtype=dtype, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    original.requires_grad_(False)
    original_capture = ResidualCapture(original, (final_layer,))
    baseline = collect_states(
        original, original_capture, validation_data, layer=final_layer,
        batches=args.geometry_batches, batch_size=min(args.batch_size, 8),
        device=args.device,
    )
    report = {
        "schema_version": 1,
        "kind": "predictor_removed_representation_evaluation",
        "checkpoint": str(args.checkpoint),
        "final_source_layer": final_layer,
        "adapted_geometry": geometry_summary(adapted),
        "original_geometry": geometry_summary(baseline),
        "centered_linear_cka_to_original": linear_cka(adapted, baseline),
        "future_retrieval": retrieval_results,
        "dataset_metadata": metadata,
        "predictor_removed": True,
        "fresh_retrieval_probes": True,
        "oracle_information": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    original_capture.__exit__(None, None, None)


if __name__ == "__main__":
    main()
