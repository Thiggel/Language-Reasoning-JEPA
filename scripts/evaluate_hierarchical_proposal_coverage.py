#!/usr/bin/env python3
"""Measure frozen-Qwen next-step proposal coverage before JEPA training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.data.igsm_step_verifier import achieved_next_state_id
from textjepa.data.provenance import sha256_file
from textjepa.utils.hierarchical_generation import (
    generate_complete_reasoning_candidates,
)
from textjepa.utils.language_planning_runtime import (
    backend_metadata,
    load_reference_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--k0", type=int, default=64)
    parser.add_argument("--max-roots", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if min(args.population, args.k0, args.max_roots) < 1:
        raise ValueError("proposal coverage sizes must be positive")
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    records = {
        str(row["problem_id"]): row
        for row in map(json.loads, args.examples.read_text().splitlines())
    }
    tokenizer, frozen = load_reference_model(args.device, args.dtype)
    roots = []
    for row, boundary_row in enumerate(features["boundaries"]):
        boundaries = boundary_row[boundary_row >= 0]
        roots.extend((row, step) for step in range(len(boundaries) - 1))
    order = torch.randperm(
        len(roots), generator=torch.Generator().manual_seed(args.seed)
    )[:args.max_roots]
    rows = []
    for root_number, root_id in enumerate(order.tolist()):
        row, step = roots[root_id]
        boundaries = features["boundaries"][row]
        boundaries = boundaries[boundaries >= 0]
        prefix = features["input_ids"][row, :int(boundaries[step])].to(
            args.device
        )
        generated = generate_complete_reasoning_candidates(
            frozen, tokenizer, prefix, population=args.population,
            max_tokens=args.k0, temperature=0.8, top_p=0.95, top_k=0,
            seed=args.seed * 100003 + root_number,
        )
        record = records[str(features["problem_id"][row])]
        valid = []
        for tokens, _ in generated:
            text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
            valid.append(achieved_next_state_id(text, record, step) >= 0)
        rows.append({
            "problem_id": str(features["problem_id"][row]),
            "boundary_index": step, "valid": valid,
        })
    metrics = {"greedy": sum(row["valid"][0] for row in rows) / len(rows)}
    for count in (1, 8, 32):
        count = min(count, args.population)
        metrics[f"oracle@{count}"] = sum(
            any(row["valid"][:count]) for row in rows
        ) / len(rows)
    payload = {
        "metrics": metrics, "rows": rows,
        "dataset_fingerprint": features["dataset_fingerprint"],
        "source_feature_sha256": sha256_file(args.features),
        "source_examples_sha256": sha256_file(args.examples),
        "backend": backend_metadata(), "population": args.population,
        "k0": args.k0, "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
