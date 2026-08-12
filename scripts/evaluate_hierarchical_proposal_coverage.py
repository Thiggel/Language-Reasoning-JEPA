#!/usr/bin/env python3
"""Measure frozen-Qwen next-step proposal coverage before JEPA training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.data.igsm_step_verifier import (
    achieved_next_state_id,
    parse_rendered_operation,
)
from textjepa.data.language_planning import (
    canonical_prompt_token_ids,
    render_igsm_solution,
)
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
    parser.add_argument(
        "--prefix-source", choices=("canonical", "self_generated"),
        default="canonical",
        help="context the worker proposes from: the teacher-forced canonical "
             "solution prefix, or the model's own free-run text",
    )
    parser.add_argument(
        "--prompt-style", choices=("pinned", "canonical"), default="pinned",
        help="'pinned' is the frozen-reference system prompt; 'canonical' is "
             "a separately labelled proposal-coverage ablation that asks for "
             "the iGSM step form (contract: Frozen reference)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def _episode_prompt(tokenizer, features, row, record, style, device):
    """Prefix the worker starts from, under the requested prompt style."""

    if style == "pinned":
        prompt_len = int(features["prompt_len"][row])
        return features["input_ids"][row, :prompt_len].to(device)
    return torch.tensor(
        canonical_prompt_token_ids(tokenizer, record["problem_text"]),
        dtype=torch.long, device=device,
    )


def _self_generated_prefix(
    frozen, tokenizer, prompt, steps, k0, seed_base
):
    """Free-run the frozen LM for ``steps`` complete steps.

    ``generate_complete_reasoning_candidates`` rejects ``temperature <= 0``, so
    near-greedy sampling stands in for argmax decoding.
    """

    prefix = prompt
    for index in range(steps):
        candidates = generate_complete_reasoning_candidates(
            frozen, tokenizer, prefix, population=1, max_tokens=k0,
            temperature=0.01, top_p=1.0, top_k=0, seed=seed_base + index,
        )
        if not candidates:
            return None
        tokens, terminal = candidates[0]
        prefix = torch.cat([prefix, tokens.to(prefix.device)])
        if terminal:
            return None
    return prefix


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
    unreachable = 0
    for root_number, root_id in enumerate(order.tolist()):
        row, step = roots[root_id]
        record = records[str(features["problem_id"][row])]
        if args.prefix_source == "canonical" and args.prompt_style == "pinned":
            boundaries = features["boundaries"][row]
            boundaries = boundaries[boundaries >= 0]
            prefix = features["input_ids"][row, :int(boundaries[step])].to(
                args.device
            )
        else:
            prompt = _episode_prompt(
                tokenizer, features, row, record, args.prompt_style,
                args.device,
            )
            if args.prefix_source == "canonical":
                # Same canonical steps, re-tokenized after the chosen prompt.
                step_lines = render_igsm_solution(
                    record["reasoning_operations"], str(record["answer"])
                )
                text = "".join(step_lines[:step])
                prefix = torch.cat([prompt, torch.tensor(
                    tokenizer.encode(text, add_special_tokens=False),
                    dtype=torch.long, device=args.device,
                )]) if step else prompt
            else:
                prefix = _self_generated_prefix(
                    frozen, tokenizer, prompt, step, args.k0,
                    args.seed * 700001 + root_number * 61,
                )
                if prefix is None:
                    # The model stopped before reaching this depth; it cannot
                    # be asked to realize the waypoint. Counted, not dropped.
                    unreachable += 1
                    rows.append({
                        "problem_id": str(features["problem_id"][row]),
                        "boundary_index": step, "valid": [],
                        "parsed": [], "unreachable": True,
                    })
                    continue
        generated = generate_complete_reasoning_candidates(
            frozen, tokenizer, prefix, population=args.population,
            max_tokens=args.k0, temperature=0.8, top_p=0.95, top_k=0,
            seed=args.seed * 100003 + root_number,
        )
        valid, parsed = [], []
        for tokens, _ in generated:
            text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
            valid.append(achieved_next_state_id(text, record, step) >= 0)
            parsed.append(parse_rendered_operation(text) is not None)
        rows.append({
            "problem_id": str(features["problem_id"][row]),
            "boundary_index": step, "valid": valid, "parsed": parsed,
            "unreachable": False,
        })
    # Unreachable roots score 0: the worker genuinely cannot realize a
    # waypoint it never arrives at, so dropping them would flatter the
    # self-generated arm exactly where it is weakest.
    metrics = {
        "greedy": sum(
            bool(row["valid"]) and row["valid"][0] for row in rows
        ) / len(rows),
        "parse_rate": sum(
            sum(row["parsed"]) / len(row["parsed"]) if row["parsed"] else 0.0
            for row in rows
        ) / len(rows),
        "unreachable_rate": unreachable / len(rows),
    }
    for count in (1, 8, 32):
        count = min(count, args.population)
        metrics[f"oracle@{count}"] = sum(
            any(row["valid"][:count]) for row in rows
        ) / len(rows)
    payload = {
        "metrics": metrics, "rows": rows,
        "prefix_source": args.prefix_source,
        "prompt_style": args.prompt_style,
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
