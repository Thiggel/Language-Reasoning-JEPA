#!/usr/bin/env python3
"""Build controlled paraphrase/semantic-contrast endpoint representations.

For TOKEN_JEPA checkpoints these are sentence-boundary ``z0`` endpoint states,
not learned sentence-level states.  ``z1`` is exported only once the checkpoint
has actually passed the sentence-JEPA training stage.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from textjepa.analysis.hierarchical_language import paired_semantic_geometry
from textjepa.data.language_planning import PAD_TOKEN_ID, prompt_token_ids
from textjepa.data.provenance import sha256_file
from textjepa.training.hierarchical_language import ResearchStage
from textjepa.utils.language_planning_runtime import (
    load_hierarchical_checkpoint,
    load_reference_model,
)


STEP = re.compile(
    r"^so the number of (?P<subject>.+?) is (?P<left>\d+) "
    r"(?P<operation>plus|minus|times) (?P<right>\d+) = "
    r"(?P<result>\d+) \.$"
)
VARIANTS = ("anchor", "paraphrase", "semantic_contrast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-igsm-groups", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


def controlled_variants(operation: str) -> tuple[str, str, str]:
    match = STEP.fullmatch(operation.strip())
    if match is None:
        raise ValueError(f"unsupported rendered iGSM operation: {operation!r}")
    fields = match.groupdict()
    result = int(fields["result"])
    wrong = (result + 1) % 23
    subject = fields["subject"]
    left, right = fields["left"], fields["right"]
    operator = fields["operation"]
    return (
        operation.rstrip("\n") + "\n",
        f"Computing the number of {subject}, {left} {operator} {right} "
        f"gives {result}.\n",
        f"Computing the number of {subject}, {left} {operator} {right} "
        f"gives {wrong}.\n",
    )


def _left_pad(sequences: list[torch.Tensor], device: str):
    width = max(map(len, sequences))
    ids = torch.full(
        (len(sequences), width), PAD_TOKEN_ID, dtype=torch.long, device=device
    )
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        ids[row, -len(sequence):] = sequence.to(device)
        mask[row, -len(sequence):] = True
    return ids, mask


def _logical_groups(tokenizer) -> list[dict]:
    prompt = torch.tensor(prompt_token_ids(
        tokenizer,
        "Write one precise declarative reasoning step about the stated class.",
    ))
    triples = (
        ("all_swans_white", "All swans are white.\n", "Every swan is white.\n", "No swan is white.\n"),
        ("no_ravens_red", "No raven is red.\n", "Not a single raven is red.\n", "Every raven is red.\n"),
        ("some_cats_black", "Some cats are black.\n", "At least one cat is black.\n", "No cat is black.\n"),
        ("all_robins_birds", "All robins are birds.\n", "Every robin is a bird.\n", "No robin is a bird.\n"),
    )
    records = []
    for name, *texts in triples:
        records.append({"name": name, "source": "logical_ood", "prefix": prompt, "texts": texts})
    return records


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.max_igsm_groups < 1 or args.batch_size < 1:
        raise ValueError("group and batch sizes must be positive")
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    examples = [json.loads(line) for line in args.examples.read_text().splitlines() if line]
    if len(examples) != len(features["problem_id"]):
        raise ValueError("feature and source-example counts differ")
    if [row["problem_id"] for row in examples] != list(features["problem_id"]):
        raise ValueError("feature and source-example ordering differs")
    tokenizer, frozen = load_reference_model(args.device, args.dtype)
    planning, learner = load_hierarchical_checkpoint(args.checkpoint, args.device)

    groups = []
    for row, example in enumerate(examples):
        boundaries = features["boundaries"][row]
        boundaries = boundaries[boundaries >= 0].tolist()
        for step, operation in enumerate(example["reasoning_operations"]):
            if len(groups) >= args.max_igsm_groups:
                break
            try:
                texts = controlled_variants(operation)
            except ValueError:
                continue
            canonical = features["input_ids"][row, boundaries[step]:boundaries[step + 1]]
            encoded = torch.tensor(tokenizer.encode(texts[0], add_special_tokens=False))
            if not torch.equal(canonical, encoded):
                raise ValueError("canonical action tokenization no longer matches feature boundaries")
            groups.append({
                "name": f'{example["problem_id"]}:{example["canonical_state_ids"][step + 1]}',
                "source": "igsm_controlled",
                "prefix": features["input_ids"][row, :boundaries[step]].clone(),
                "texts": texts,
            })
        if len(groups) >= args.max_igsm_groups:
            break
    groups.extend(_logical_groups(tokenizer))
    sequences, metadata = [], []
    for group_id, group in enumerate(groups):
        for variant_id, text in enumerate(group["texts"]):
            action = torch.tensor(tokenizer.encode(text, add_special_tokens=False))
            sequences.append(torch.cat([group["prefix"], action]))
            metadata.append({
                "group_id": group_id,
                "group": group["name"],
                "source": group["source"],
                "variant_id": variant_id,
                "variant": VARIANTS[variant_id],
                "text": text.rstrip("\n"),
            })
    hidden = []
    for start in range(0, len(sequences), args.batch_size):
        ids, mask = _left_pad(sequences[start:start + args.batch_size], args.device)
        output = frozen(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            logits_to_keep=1,
        )
        hidden.append(output.hidden_states[-1][:, -1].float().cpu())
    h = torch.cat(hidden)
    dtype = next(planning.parameters()).dtype
    z0 = planning.e0(h.to(args.device, dtype=dtype)).float().cpu()
    z0_device = z0.to(args.device)
    z0_whitened = learner.token_metric.whiten(
        z0_device - learner.token_metric.mean
    ).float().cpu()
    representations = {
        "h": h,
        "z0_sentence_boundary": z0,
        "z0_mahalanobis_whitened": z0_whitened,
    }
    if learner.stage >= ResearchStage.SENTENCE_JEPA:
        z1 = planning.e0_to_1(
            z0.to(args.device, dtype=dtype)
        ).float()
        representations["z1_sentence"] = z1.cpu()
        representations["z1_mahalanobis_whitened"] = (
            learner.sentence_metric.whiten(
                z1 - learner.sentence_metric.mean
            ).float().cpu()
        )
    group_ids = torch.tensor([row["group_id"] for row in metadata])
    variant_ids = torch.tensor([row["variant_id"] for row in metadata])
    metrics = {
        name: paired_semantic_geometry(states, group_ids, variant_ids)
        for name, states in representations.items()
    }
    payload = {
        "representations": representations,
        "group_ids": group_ids,
        "variant_ids": variant_ids,
        "metadata": metadata,
        "metrics": metrics,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset_fingerprint": features["dataset_fingerprint"],
        "checkpoint_stage": learner.stage.name,
        "interpretation": (
            "z0 entries are exact sentence-boundary token states; z1 is "
            "included only for sentence-JEPA-or-later checkpoints"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
