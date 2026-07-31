#!/usr/bin/env python3
"""Materialize contextual macro rollouts toward verified iGSM terminals."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from textjepa.data.language_planning import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_VERSION,
)
from textjepa.data.provenance import artifact_fingerprint, sha256_file
from textjepa.models.hierarchical_language_jepa import (
    HIERARCHICAL_LANGUAGE_ARCHITECTURE,
)
from textjepa.planning.nested_language_runtime import (
    rollout_prior_noise,
    sentence_planning_state_from_trace,
)
from textjepa.training.hierarchical_language import ResearchStage
from textjepa.utils.language_planning_runtime import (
    load_hierarchical_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-roots", type=int, default=256)
    parser.add_argument("--first-actions", type=int, default=16)
    parser.add_argument("--continuation-samples", type=int, default=4)
    parser.add_argument("--max-prefix", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _valid_boundaries(row: torch.Tensor) -> torch.Tensor:
    result = row[row >= 0]
    if len(result) < 2 or bool((result[1:] <= result[:-1]).any()):
        raise ValueError("feature row has invalid reasoning boundaries")
    return result


def _root_catalog(features: dict, maximum: int, seed: int) -> list[tuple[int, int]]:
    roots = []
    for row in range(len(features["hidden_states"])):
        boundaries = _valid_boundaries(features["boundaries"][row])
        roots.extend((row, index) for index in range(len(boundaries) - 1))
    if not roots:
        raise ValueError("feature artifact has no nonterminal sentence roots")
    if len(roots) > maximum:
        order = torch.randperm(
            len(roots), generator=torch.Generator().manual_seed(seed)
        )[:maximum]
        roots = [roots[int(index)] for index in order]
    return roots


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if min(
        args.max_roots, args.first_actions,
        args.continuation_samples, args.max_prefix,
    ) < 1:
        raise ValueError("oracle rollout sizes must be positive")
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    required = {
        "hidden_states", "input_ids", "boundaries", "prompt_len",
        "solution_end", "problem_id", "reasoning_depth",
        "canonical_state_ids", "symbolically_verified",
        "dataset_fingerprint", "model_id", "model_revision",
        "transformers_version",
    }
    if not required <= features.keys():
        raise ValueError("oracle rollouts require canonical verified features")
    if (
        features["model_id"] != MODEL_ID
        or features["model_revision"] != MODEL_REVISION
        or features["transformers_version"] != TRANSFORMERS_VERSION
        or not all(features["symbolically_verified"])
    ):
        raise ValueError("oracle rollout feature provenance is incompatible")
    model, learner = load_hierarchical_checkpoint(
        args.checkpoint, args.device
    )
    if learner.stage != ResearchStage.MACRO_ACTION or model.pi1 is None:
        raise ValueError("oracle rollouts require a MACRO_ACTION checkpoint")
    dtype = next(model.parameters()).dtype
    roots = _root_catalog(features, args.max_roots, args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 91)
    rollout_states, rollout_logp = [], []
    successors, successor_contexts, task_hidden_rows, goals = [], [], [], []
    metadata = []
    first, samples, depth = (
        args.first_actions, args.continuation_samples, args.max_prefix
    )
    for row, boundary_index in roots:
        boundaries = _valid_boundaries(features["boundaries"][row]).to(
            args.device
        )
        hidden = features["hidden_states"][row].to(
            args.device, dtype=dtype
        )
        ids = features["input_ids"][row].to(args.device)
        planning_state = sentence_planning_state_from_trace(
            model, hidden, ids, boundaries, boundary_index
        )
        prompt = int(features["prompt_len"][row])
        task_hidden = hidden[prompt - 1]
        task = model.task_projection(task_hidden)
        first_noise = torch.randn(
            first, model.config.d_action,
            device=args.device, dtype=dtype, generator=generator,
        )
        noise = torch.randn(
            first, samples, depth, model.config.d_action,
            device=args.device, dtype=dtype, generator=generator,
        )
        noise[:, :, 0] = first_noise[:, None]
        rollout = rollout_prior_noise(
            model, planning_state, task,
            noise.reshape(first * samples, depth, model.config.d_action),
        )
        states = rollout.states.reshape(
            first, samples, depth, model.config.d_sentence
        )
        logp = rollout.log_probabilities.reshape(first, samples, depth)
        rollout_states.append(states.cpu())
        rollout_logp.append(logp.cpu())
        successors.append(states[:, 0, 0].cpu())
        successor_contexts.append(
            rollout.contexts.reshape(
                first, samples, depth, model.config.predictor_width
            )[:, 0, 0].cpu()
        )
        task_hidden_rows.append(task_hidden.cpu())
        terminal_prefix = int(features["solution_end"][row])
        terminal = model.encode_sentence(
            hidden[terminal_prefix - 1], target=True
        )
        goals.append(terminal[None].cpu())
        metadata.append({
            "problem_id": str(features["problem_id"][row]),
            "root_boundary_index": boundary_index,
            "root_prefix_length": int(boundaries[boundary_index]),
            "symbolic_depth": int(features["reasoning_depth"][row]),
        })
    state_tensor = torch.stack(rollout_states)
    logp_tensor = torch.stack(rollout_logp)
    goal_tensor = torch.stack(goals)
    payload = {
        "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_feature_sha256": sha256_file(args.features),
        "dataset_fingerprint": features["dataset_fingerprint"],
        "rollout_states": state_tensor,
        "rollout_log_probabilities": logp_tensor,
        "rollout_mask": torch.ones_like(logp_tensor, dtype=torch.bool),
        "goals": goal_tensor,
        "goal_mask": torch.ones(
            len(goals), 1, dtype=torch.bool
        ),
        "successor_state": torch.stack(successors),
        "successor_context": torch.stack(successor_contexts),
        "task_hidden": torch.stack(task_hidden_rows),
        "first_action_log_probability": logp_tensor[:, :, 0, 0],
        "action_mask": torch.ones(
            len(roots), first, dtype=torch.bool
        ),
        "metadata": metadata,
        "symbolically_verified": True,
        "sampling": {
            "first_actions": first,
            "continuation_samples": samples,
            "total_prefixes": depth,
            "seed": args.seed,
            "prior_coordinate_sampling": True,
            "context_history_preserved": True,
        },
    }
    payload["terminal_set_fingerprint"] = artifact_fingerprint(
        payload, ("goals", "goal_mask", "metadata")
    )
    payload["oracle_rollout_fingerprint"] = artifact_fingerprint(
        payload, (
            "rollout_states", "rollout_log_probabilities", "rollout_mask",
            "successor_state", "successor_context", "task_hidden",
            "first_action_log_probability", "action_mask", "metadata",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)


if __name__ == "__main__":
    main()
