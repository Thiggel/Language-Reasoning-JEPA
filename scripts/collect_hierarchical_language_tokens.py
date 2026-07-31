#!/usr/bin/env python3
"""Materialize the canonical Qwen token/indexing artifact without LM forwards."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from collect_hierarchical_language_features import read_examples
from textjepa.data.language_planning import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_VERSION,
    collate_language_planning_examples,
)
from textjepa.data.provenance import artifact_fingerprint, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    examples = read_examples(args.input, tokenizer)
    payload = collate_language_planning_examples(examples)
    payload.update({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "input_fingerprint": sha256_file(args.input),
        "problem_id": [example.problem_id for example in examples],
        "template_family": [
            example.template_family for example in examples
        ],
        "graph_family": [example.graph_family for example in examples],
        "symbolically_verified": [
            example.symbolically_verified for example in examples
        ],
    })
    payload["dataset_fingerprint"] = artifact_fingerprint(payload, (
        "model_id", "model_revision", "transformers_version",
        "input_fingerprint", "input_ids", "attention_mask",
        "prompt_len", "solution_end", "boundaries", "reasoning_depth",
        "canonical_state_ids", "problem_id", "template_family",
        "graph_family", "symbolically_verified",
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
