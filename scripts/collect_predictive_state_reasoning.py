#!/usr/bin/env python3
"""Extract boundary states from externally verified reasoning trajectories.

Input JSONL records require `problem_id`, `prompt`, `trajectory`, and `correct`.
The verifier that produced `correct` remains part of dataset provenance; this
collector never infers correctness from latent state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from textjepa.data.predictive_state import (
    REASONING_STATE_SCHEMA,
    sha256_path,
)
from textjepa.models.action_transition import (
    ResidualCapture,
    decoder_layers,
    parameter_free_rms_norm,
)
from textjepa.training.predictive_state import (
    architecture_defaults,
    load_stage1_checkpoint,
    load_stage2_checkpoint,
    teacher_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--source-layer", type=int, action="append")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def load(args, dtype):
    payload = None
    if args.checkpoint:
        header = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if header.get("kind") == "action_conditioned_cross_layer_stage2":
            model, predictor, payload = load_stage2_checkpoint(
                args.checkpoint, device=args.device, dtype=dtype
            )
        else:
            model, predictor, payload = load_stage1_checkpoint(
                args.checkpoint, device=args.device, dtype=dtype
            )
        model_id = payload.get("model_id", payload.get("stage1_payload", {}).get("model_id"))
        revision = payload.get("model_revision", payload.get("stage1_payload", {}).get("model_revision"))
        if args.source_layer:
            sources = tuple(args.source_layer)
        elif predictor is not None:
            sources = tuple(predictor.config.source_layers)
        else:
            sources = tuple(
                architecture_defaults(model_id, len(decoder_layers(model)))["sources"]
            )
    else:
        if not args.model_id or not args.model_revision:
            raise ValueError("original model extraction needs model ID and revision")
        model_id, revision = args.model_id, args.model_revision
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(args.device)
        defaults = architecture_defaults(model_id, len(decoder_layers(model)))
        sources = tuple(args.source_layer or defaults["sources"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    return model.eval(), tokenizer, model_id, revision, sources, payload


def boundary_positions(text: str, offsets: list[tuple[int, int]],
                       prompt_characters: int) -> list[int]:
    positions = []
    for token_index, (_begin, end) in enumerate(offsets):
        if end <= prompt_characters or end < 1:
            continue
        if text[end - 1] in "\n.!?":
            positions.append(token_index)
    if not positions or positions[-1] != len(offsets) - 1:
        positions.append(len(offsets) - 1)
    return sorted(set(positions))


def prompt_state_index(
    offsets: list[tuple[int, int]], prompt_characters: int
) -> int:
    """Find the last token wholly contained in the prompt text.

    Deriving this from the joint prompt-plus-trajectory encoding avoids BPE
    boundary merges that make a separately tokenized prompt length invalid.
    """
    contained = [
        index for index, (begin, end) in enumerate(offsets)
        if end > begin and end <= prompt_characters
    ]
    if not contained:
        raise ValueError("joint tokenization contains no complete prompt token")
    return contained[-1]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    model, tokenizer, model_id, revision, sources, checkpoint = load(args, dtype)
    capture = ResidualCapture(model, sources)
    records = []
    for line_number, line in enumerate(
        args.input.read_text(encoding="utf-8").splitlines(), 1
    ):
        raw = json.loads(line)
        required = {"problem_id", "prompt", "trajectory", "correct"}
        if not required <= raw.keys():
            raise ValueError(f"line {line_number} lacks required trajectory fields")
        prompt = str(raw["prompt"])
        separator = "" if prompt.endswith((" ", "\n")) else "\n"
        text = prompt + separator + str(raw["trajectory"])
        encoded = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(args.device)
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"][0]]
        output = teacher_forward(
            model, ids, capture=capture, attention_mask=None, use_cache=False
        )
        prompt_index = prompt_state_index(offsets, len(prompt + separator))
        positions = boundary_positions(text, offsets, len(prompt + separator))
        fused = torch.cat([
            parameter_free_rms_norm(output.states[layer][0])
            for layer in sources
        ], dim=-1).cpu()
        selected = fused[positions]
        length = len(selected)
        record = {
            "problem_id": str(raw["problem_id"]),
            "trajectory_id": str(raw.get("trajectory_id", f"line-{line_number}")),
            "prompt_state": fused[prompt_index],
            "states": selected,
            "valid": torch.ones(length, dtype=torch.bool),
            "remaining_chunks": torch.arange(length - 1, -1, -1, dtype=torch.float32),
            "correct": bool(raw["correct"]),
            "terminal_index": length - 1,
            "answer": raw.get("answer"),
            "verifier": raw.get("verifier"),
            "boundary_token_indices": torch.tensor(positions),
        }
        records.append(record)
    metadata = {
        "model_id": model_id,
        "model_revision": revision,
        "checkpoint_fingerprint": (
            sha256_path(args.checkpoint) if args.checkpoint else f"original:{revision}"
        ),
        "source_layers": list(sources),
        "input_sha256": sha256_path(args.input),
        "oracle_terminal_states": True,
        "candidate_privileged_outcomes": True,
        "cross_project_information": False,
        "correctness_source": "external_verified_input",
        "boundary": "newline_or_sentence_punctuation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": REASONING_STATE_SCHEMA,
        "metadata": metadata,
        "records": records,
    }, args.output)
    print(json.dumps(metadata | {"records": len(records)}), flush=True)
    capture.__exit__(None, None, None)


if __name__ == "__main__":
    main()
