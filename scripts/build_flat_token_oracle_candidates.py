#!/usr/bin/env python3
"""Generate and ground flat token-space oracle candidate populations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.analysis.compute import ComputeLedger, inference_flops
from textjepa.data.language_planning import (
    GENERATION_EOS_TOKEN_IDS,
    MAX_STEP_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    PAD_TOKEN_ID,
    TRANSFORMERS_VERSION,
)
from textjepa.data.provenance import artifact_fingerprint, sha256_file
from textjepa.models.hierarchical_language_jepa import (
    HIERARCHICAL_LANGUAGE_ARCHITECTURE,
)
from textjepa.training.hierarchical_language import ResearchStage
from textjepa.utils.language_planning_runtime import (
    backend_metadata as _backend_metadata,
    load_hierarchical_checkpoint as load_checkpoint,
    load_reference_model,
    text_parameter_count as _text_parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-split", required=True)
    parser.add_argument("--method-label", default="token_jepa")
    parser.add_argument("--horizon", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--max-roots", type=int, default=256)
    parser.add_argument("--root-batch-size", type=int, default=4)
    parser.add_argument("--reencode-batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


def _root_records(features: dict, horizon: int, maximum: int) -> list[dict]:
    roots = []
    for row in range(len(features["hidden_states"])):
        prompt = int(features["prompt_len"][row])
        end = int(features["solution_end"][row])
        boundaries = features["boundaries"][row]
        for root in boundaries[boundaries >= 0].tolist()[:-1]:
            root = int(root)
            if root >= prompt and root + horizon <= end:
                roots.append({
                    "row": row,
                    "root": root,
                    "prompt": prompt,
                    "end": end,
                })
    generator = torch.Generator().manual_seed(1729 + horizon + maximum)
    if len(roots) > maximum:
        order = torch.randperm(len(roots), generator=generator)[:maximum]
        roots = [roots[int(index)] for index in order]
    if not roots:
        raise ValueError("feature artifact has no roots supporting this horizon")
    return roots


def _left_pad(sequences: list[torch.Tensor], device: str):
    width = max(len(sequence) for sequence in sequences)
    ids = torch.full(
        (len(sequences), width), PAD_TOKEN_ID,
        dtype=torch.long, device=device,
    )
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        ids[row, width - len(sequence):] = sequence.to(device)
        mask[row, width - len(sequence):] = True
    return ids, mask


@torch.no_grad()
def _ground_candidates(
    frozen_model,
    prefixes: list[torch.Tensor],
    candidates: torch.Tensor,
    *,
    device: str,
    batch_size: int,
    ledger: ComputeLedger,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_prefixes, flat_candidates = [], []
    for prefix, population in zip(prefixes, candidates):
        for candidate in population:
            flat_prefixes.append(prefix)
            flat_candidates.append(candidate.cpu())
    endpoints, log_probabilities = [], []
    parameters = _text_parameter_count(frozen_model)
    horizon = candidates.shape[-1]
    for start in range(0, len(flat_prefixes), batch_size):
        prefix_chunk = flat_prefixes[start:start + batch_size]
        candidate_chunk = flat_candidates[start:start + batch_size]
        sequences = [
            torch.cat([prefix, candidate])
            for prefix, candidate in zip(prefix_chunk, candidate_chunk)
        ]
        ids, mask = _left_pad(sequences, device)
        with ledger.measure(
            "flat_oracle_exact_reencoding",
            estimated_flops=inference_flops(
                parameters, int(mask.sum())
            ),
            items=len(sequences),
        ):
            output = frozen_model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
                logits_to_keep=horizon + 1,
            )
        endpoints.append(output.hidden_states[-1][:, -1].cpu())
        logits = output.logits[:, -(horizon + 1):-1].float()
        token = torch.stack(candidate_chunk).to(device)
        log_probability = torch.log_softmax(logits, -1).gather(
            -1, token[..., None]
        ).squeeze(-1).sum(-1)
        log_probabilities.append(log_probability.cpu())
    shape = candidates.shape[:2]
    return (
        torch.cat(endpoints).reshape(*shape, -1),
        torch.cat(log_probabilities).reshape(shape),
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.population < 2 or args.max_roots < 1:
        raise ValueError("population and roots must be positive")
    if args.top_k < 0 or not 0 < args.top_p <= 1 or args.temperature <= 0:
        raise ValueError("sampling configuration is invalid")
    features = torch.load(
        args.features, map_location="cpu", weights_only=True
    )
    required = {
        "hidden_states", "input_ids", "prompt_len", "solution_end",
        "boundaries", "reasoning_depth", "problem_id",
        "dataset_fingerprint", "model_id", "model_revision",
        "transformers_version",
    }
    if not required <= features.keys():
        raise ValueError("flat oracle requires canonical collected features")
    if (
        features["model_id"] != MODEL_ID
        or features["model_revision"] != MODEL_REVISION
        or features["transformers_version"] != TRANSFORMERS_VERSION
    ):
        raise ValueError("feature provenance is incompatible")
    planning_model, learner = load_checkpoint(args.checkpoint, args.device)
    if learner.stage != ResearchStage.TOKEN_JEPA:
        raise ValueError("flat oracle requires a TOKEN_JEPA checkpoint")
    tokenizer, frozen_model = load_reference_model(args.device, args.dtype)
    del tokenizer
    roots = _root_records(features, args.horizon, args.max_roots)
    ledger = ComputeLedger()
    candidate_chunks = []
    for start in range(0, len(roots), args.root_batch_size):
        chunk = roots[start:start + args.root_batch_size]
        prefixes = [
            features["input_ids"][item["row"], :item["root"]]
            for item in chunk
        ]
        ids, mask = _left_pad(prefixes, args.device)
        common = {
            "max_new_tokens": args.horizon,
            "min_new_tokens": args.horizon,
            "eos_token_id": None,
            "pad_token_id": PAD_TOKEN_ID,
            "attention_mask": mask,
        }
        parameters = _text_parameter_count(frozen_model)
        with ledger.measure(
            "flat_oracle_proposal_generation",
            estimated_flops=inference_flops(
                parameters,
                int(mask.sum()) + len(chunk) * args.horizon
                + len(chunk) * (args.population - 1) * args.horizon,
            ),
            items=len(chunk) * args.population,
        ):
            greedy = frozen_model.generate(
                ids, do_sample=False, **common
            )[:, -args.horizon:]
            cuda_devices = []
            if ids.device.type == "cuda":
                cuda_devices = [ids.device.index or 0]
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(args.seed + start)
                sampled = frozen_model.generate(
                    ids,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    num_return_sequences=args.population - 1,
                    **common,
                )[:, -args.horizon:]
        sampled = sampled.reshape(
            len(chunk), args.population - 1, args.horizon
        )
        candidate_chunks.append(torch.cat([
            greedy[:, None], sampled
        ], 1).cpu())
    candidates = torch.cat(candidate_chunks)
    prefixes = [
        features["input_ids"][item["row"], :item["root"]].clone()
        for item in roots
    ]
    exact_hidden, log_probability = _ground_candidates(
        frozen_model, prefixes, candidates,
        device=args.device, batch_size=args.reencode_batch_size,
        ledger=ledger,
    )
    planning_dtype = next(planning_model.parameters()).dtype
    exact_endpoint = planning_model.e0(
        exact_hidden.to(args.device, dtype=planning_dtype)
    ).cpu()
    predicted = []
    waypoint = []
    reference = []
    for item, population in zip(roots, candidates):
        row, root, prompt = item["row"], item["root"], item["prompt"]
        first_prefix = max(
            prompt, root - planning_model.config.token_context + 1
        )
        prefix_lengths = torch.arange(first_prefix, root + 1)
        state_history = planning_model.e0(
            features["hidden_states"][
                row, prefix_lengths - 1
            ].to(args.device, dtype=planning_dtype)
        )
        action_history = planning_model.token_action(
            features["input_ids"][
                row, first_prefix:root
            ].to(args.device)
        )
        action = planning_model.token_action(
            population.to(args.device)
        )
        rollout, _ = planning_model.p0.rollout(
            state_history[-1], action,
            state_history=state_history[None],
            action_history=action_history[None],
        )
        predicted.append(rollout[:, -1].cpu())
        waypoint.append(planning_model.e0_target(
            features["hidden_states"][
                row, root + args.horizon - 1
            ].to(args.device, dtype=planning_dtype)
        ).cpu())
        reference.append(features["input_ids"][
            row, root:root + args.horizon
        ])
    metadata = [{
        "dataset_split": args.dataset_split,
        "symbolic_depth": int(features["reasoning_depth"][item["row"]]),
        "population_size": args.population,
        "token_oracle_horizon": args.horizon,
        "endpoint_kind": "paired",
        "planning_effort_candidate_tokens": args.population * args.horizon,
        "problem_id": str(features["problem_id"][item["row"]]),
        "root_prefix_length": item["root"],
        "method_label": args.method_label,
    } for item in roots]
    payload = {
        "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset_fingerprint": features["dataset_fingerprint"],
        "source_feature_sha256": sha256_file(args.features),
        "candidate_tokens": candidates,
        "candidate_log_probability": log_probability,
        "predicted_endpoints": torch.stack(predicted),
        "exact_endpoints": exact_endpoint,
        "waypoint": torch.stack(waypoint),
        "reference_tokens": torch.stack(reference),
        "metadata": metadata,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "greedy_candidates": 1,
            "sampled_candidates": args.population - 1,
            "thinking_mode": False,
            "generation_stop_ids_disabled_for_fixed_horizon": list(
                GENERATION_EOS_TOKEN_IDS
            ),
        },
        "backend": _backend_metadata(),
        "compute": ledger.summary(),
    }
    fields = (
        "candidate_tokens", "candidate_log_probability",
        "predicted_endpoints", "exact_endpoints", "waypoint",
        "reference_tokens", "metadata",
    )
    payload["candidate_payload_fingerprint"] = artifact_fingerprint(
        payload, fields
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)


if __name__ == "__main__":
    main()
