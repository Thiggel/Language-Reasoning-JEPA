#!/usr/bin/env python3
"""Tokenize an ordinary-language corpus into boundary-safe fixed blocks."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import urllib.request

from transformers import AutoTokenizer

from textjepa.data.predictive_state import (
    pack_documents,
    save_token_blocks,
    sha256_path,
    split_whitespace_documents,
    text_fingerprint,
)
from textjepa.training.predictive_state import require_transformers_runtime


QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B"
QWEN_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
WIKITEXT2_TRAIN_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2/train.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--url", default=WIKITEXT2_TRAIN_URL)
    parser.add_argument("--download-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=QWEN_MODEL_ID)
    parser.add_argument("--model-revision", default=QWEN_REVISION)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_documents(path: Path) -> list[str]:
    # WikiText uses whitespace-only lines between paragraphs (often a single
    # space, not a literal empty line). Keeping headings as their own documents
    # makes boundary handling explicit and reproducible.
    return split_whitespace_documents(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    require_transformers_runtime()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation fraction must lie in (0, 0.5)")
    source = args.input
    if source is None:
        source = args.download_path or args.output.with_suffix(".source.txt")
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            urllib.request.urlretrieve(args.url, source)
    if not source.is_file():
        raise FileNotFoundError(source)
    documents = read_documents(source)
    if args.max_documents is not None:
        documents = documents[:args.max_documents]
    if len(documents) < 2:
        raise ValueError("corpus must contain at least two documents")
    order = list(range(len(documents)))
    random.Random(args.seed).shuffle(order)
    validation_count = max(1, round(len(order) * args.validation_fraction))
    validation_indices = set(order[:validation_count])
    train_text = [text for i, text in enumerate(documents)
                  if i not in validation_indices]
    validation_text = [text for i, text in enumerate(documents)
                       if i in validation_indices]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision, use_fast=True
    )
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no EOS token")

    def encode(texts: list[str]) -> list[list[int]]:
        result = []
        for begin in range(0, len(texts), 256):
            encoded = tokenizer(
                texts[begin:begin + 256], add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            result.extend(encoded)
        return result

    train = pack_documents(
        encode(train_text), context_length=args.context_length,
        eos_token_id=eos,
    )
    validation = pack_documents(
        encode(validation_text), context_length=args.context_length,
        eos_token_id=eos,
    )
    metadata = {
        "source_path": str(source.resolve()),
        "source_url": None if args.input else args.url,
        "source_sha256": sha256_path(source),
        "document_fingerprint": text_fingerprint(documents),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "eos_token_id": eos,
        "context_length": args.context_length,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "train_documents": len(train_text),
        "validation_documents": len(validation_text),
        "train_blocks": len(train["input_ids"]),
        "validation_blocks": len(validation["input_ids"]),
        "boundary_safe_target_mask": True,
        "ordinary_language_only": True,
        "oracle_information": False,
        "candidate_privileged_information": False,
        "cross_project_information": False,
    }
    save_token_blocks(
        args.output, train=train, validation=validation, metadata=metadata
    )
    print(metadata, flush=True)


if __name__ == "__main__":
    main()
