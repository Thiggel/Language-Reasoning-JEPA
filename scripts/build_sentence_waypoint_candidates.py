#!/usr/bin/env python3
"""Build and symbolically ground the oracle next-sentence worker gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from textjepa.data.igsm_step_verifier import achieved_next_state_id
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
    advance_sentence_planning_state,
    sentence_planning_state_from_trace,
)
from textjepa.training.hierarchical_language import ResearchStage
from textjepa.utils.hierarchical_generation import (
    exact_ground_sentence_candidates,
    generate_complete_reasoning_candidates,
)
from textjepa.utils.language_planning_runtime import (
    backend_metadata,
    load_hierarchical_checkpoint,
    load_reference_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-split", required=True)
    parser.add_argument("--k0", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--max-roots", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def _load_examples(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            problem_id = str(record["problem_id"])
            if problem_id in rows:
                raise ValueError("duplicate problem ID in iGSM examples")
            rows[problem_id] = record
    return rows


def _boundaries(row: torch.Tensor) -> torch.Tensor:
    result = row[row >= 0]
    if len(result) < 2 or bool((result[1:] <= result[:-1]).any()):
        raise ValueError("invalid global prefix boundaries")
    return result


def _roots(features: dict, maximum: int, seed: int) -> list[tuple[int, int]]:
    roots = []
    for row in range(len(features["hidden_states"])):
        roots.extend(
            (row, index)
            for index in range(len(_boundaries(features["boundaries"][row])) - 1)
        )
    if len(roots) > maximum:
        order = torch.randperm(
            len(roots), generator=torch.Generator().manual_seed(seed)
        )[:maximum]
        roots = [roots[int(index)] for index in order]
    if not roots:
        raise ValueError("sentence-worker feature set has no roots")
    return roots


def _effective_rank(states: torch.Tensor) -> float:
    centered = states.float() - states.float().mean(0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    denominator = float(eigenvalues.square().sum())
    return float(eigenvalues.sum().square() / denominator) if denominator else 0.0


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if min(args.population, args.max_roots) < 2:
        raise ValueError("worker population and roots must be at least two")
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    required = {
        "hidden_states", "input_ids", "boundaries", "prompt_len",
        "solution_end", "problem_id", "reasoning_depth",
        "canonical_state_ids", "symbolically_verified",
        "dataset_fingerprint", "model_id", "model_revision",
        "transformers_version",
    }
    if not required <= features.keys() or not all(
        features.get("symbolically_verified", [])
    ):
        raise ValueError("worker gate requires verified canonical features")
    if (
        features["model_id"] != MODEL_ID
        or features["model_revision"] != MODEL_REVISION
        or features["transformers_version"] != TRANSFORMERS_VERSION
    ):
        raise ValueError("worker feature provenance is incompatible")
    examples = _load_examples(args.examples)
    model, learner = load_hierarchical_checkpoint(args.checkpoint, args.device)
    if learner.stage not in {
        ResearchStage.SENTENCE_JEPA, ResearchStage.DYNAMIC_COMMUTATION,
    }:
        raise ValueError(
            "worker candidates require a sentence or commutation checkpoint"
        )
    tokenizer, frozen = load_reference_model(args.device, args.dtype)
    dtype = next(model.parameters()).dtype
    roots = _roots(features, args.max_roots, args.seed)
    candidate_tokens, candidate_lengths, candidate_logp = [], [], []
    predicted_endpoints, exact_endpoints, waypoints = [], [], []
    symbolic_ids, waypoint_ids, metadata = [], [], []
    true_dynamics, identity_dynamics, commutation_errors = [], [], []
    boundary_states = []
    for root_number, (row, boundary_index) in enumerate(roots):
        problem_id = str(features["problem_id"][row])
        if problem_id not in examples:
            raise ValueError("feature/example problem IDs do not align")
        boundaries = _boundaries(features["boundaries"][row]).to(args.device)
        hidden = features["hidden_states"][row].to(args.device, dtype=dtype)
        ids = features["input_ids"][row].to(args.device)
        root = int(boundaries[boundary_index])
        next_root = int(boundaries[boundary_index + 1])
        reference = ids[root:next_root]
        generated = generate_complete_reasoning_candidates(
            frozen, tokenizer, ids[:root], population=args.population,
            max_tokens=args.k0, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k,
            seed=args.seed * 100003 + root_number,
            reference=reference,
        )
        grounded = exact_ground_sentence_candidates(
            frozen, ids[:root], generated
        )
        exact_token = model.e0(grounded.endpoint_hidden.to(dtype=dtype))
        exact_endpoints.append(exact_token.cpu())
        planning_state = sentence_planning_state_from_trace(
            model, hidden, ids, boundaries, boundary_index
        )
        first_prefix = max(
            int(features["prompt_len"][row]),
            root - model.config.token_context + 1,
        )
        prefix_lengths = torch.arange(
            first_prefix, root + 1, device=args.device
        )
        token_history = model.e0(hidden[prefix_lengths - 1])[None]
        token_action_history = model.token_action(
            ids[first_prefix:root]
        )[None]
        predicted = []
        for candidate, _ in generated:
            action = model.token_action(candidate.to(args.device))[None]
            rollout, _ = model.p0.rollout(
                token_history[0, -1], action,
                state_history=token_history,
                action_history=token_action_history,
            )
            predicted.append(rollout[0, -1])
        predicted_endpoints.append(torch.stack(predicted).cpu())
        waypoint = model.encode_sentence(hidden[next_root - 1], target=True)
        waypoints.append(waypoint.cpu())
        padded_tokens = torch.full(
            (args.population, args.k0), model.config.pad_id,
            dtype=torch.long,
        )
        padded_tokens[:, :grounded.tokens.shape[1]] = grounded.tokens.cpu()
        candidate_tokens.append(padded_tokens)
        candidate_lengths.append(grounded.lengths.cpu())
        candidate_logp.append(grounded.log_probabilities.cpu())
        decoded = [
            tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
            for tokens, _ in generated
        ]
        achieved = torch.tensor([
            achieved_next_state_id(text, examples[problem_id], boundary_index)
            for text in decoded
        ], dtype=torch.long)
        symbolic_ids.append(achieved)
        goal_id = int(features["canonical_state_ids"][row, boundary_index + 1])
        waypoint_ids.append(goal_id)
        # Held-out true one-step sentence dynamics uses the complete cache.
        width = len(reference)
        observed_action = model.a1(
            reference[None], torch.ones(
                1, width, dtype=torch.bool, device=args.device
            )
        )
        predicted_true = advance_sentence_planning_state(
            model, planning_state, observed_action
        ).state[0]
        true_dynamics.append(float(learner.sentence_metric(
            predicted_true[None], waypoint[None]
        )[0]))
        identity_dynamics.append(float(learner.sentence_metric(
            planning_state.state, waypoint[None]
        )[0]))
        # Candidate zero is the exact observed reference injected above.
        predicted_reference_coarse = model.e0_to_1(
            predicted[0][None]
        )
        commutation_errors.append(float(learner.sentence_metric(
            predicted_reference_coarse, predicted_true[None]
        )[0]))
        boundary_states.extend([
            model.encode_sentence(hidden[position - 1], target=True).cpu()
            for position in boundaries.tolist()
        ])
        metadata.append({
            "dataset_split": args.dataset_split,
            "symbolic_depth": int(features["reasoning_depth"][row]),
            "population_size": args.population,
            "k0": args.k0,
            "endpoint_kind": "paired",
            "problem_id": problem_id,
            "root_prefix_length": root,
            "root_boundary_index": boundary_index,
            "planning_effort_candidate_tokens": args.population * args.k0,
        })
    tokens = torch.stack(candidate_tokens)
    logp = torch.stack(candidate_logp)
    predicted = torch.stack(predicted_endpoints)
    exact = torch.stack(exact_endpoints)
    waypoint = torch.stack(waypoints)
    symbolic = torch.stack(symbolic_ids)
    waypoint_symbolic = torch.tensor(waypoint_ids)
    # Geometry purity asks whether exact latent selection identifies a
    # verifier-approved successor; it cannot be satisfied by latent collapse.
    exact_cost = learner.sentence_metric(
        model.e0_to_1(exact.to(args.device, dtype=dtype)),
        waypoint.to(args.device, dtype=dtype)[:, None],
    ).cpu()
    selected = exact_cost.argmin(-1)
    purity = float((
        symbolic[torch.arange(len(symbolic)), selected]
        == waypoint_symbolic
    ).float().mean())
    mean_dynamics = sum(true_dynamics) / len(true_dynamics)
    mean_identity = sum(identity_dynamics) / len(identity_dynamics)
    metrics = {
        "heldout_sentence_dynamics": mean_dynamics,
        "heldout_sentence_identity": mean_identity,
        "heldout_sentence_dynamics_gain": mean_identity - mean_dynamics,
        "heldout_commutation_error": (
            sum(commutation_errors) / len(commutation_errors)
        ),
        "sentence_effective_rank": _effective_rank(
            torch.stack(boundary_states)
        ),
        "symbolic_state_purity": purity,
    }
    passed = (
        all(math.isfinite(value) for value in metrics.values())
        and metrics["heldout_sentence_dynamics_gain"] > 0.0
        and metrics["sentence_effective_rank"] >= 4.0
        and metrics["symbolic_state_purity"] > 0.0
    )
    payload = {
        "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_feature_sha256": sha256_file(args.features),
        "dataset_fingerprint": features["dataset_fingerprint"],
        "candidate_tokens": tokens,
        "candidate_lengths": torch.stack(candidate_lengths),
        "candidate_log_probability": logp,
        "predicted_endpoints": predicted,
        "exact_endpoints": exact,
        "waypoint": waypoint,
        "candidate_boundary_complete": torch.ones_like(symbolic, dtype=torch.bool),
        "candidate_symbolic_state_id": symbolic,
        "waypoint_symbolic_state_id": waypoint_symbolic,
        "metadata": metadata,
        "backend": backend_metadata(),
    }
    fields = (
        "candidate_tokens", "candidate_log_probability",
        "predicted_endpoints", "exact_endpoints", "waypoint",
        "candidate_boundary_complete", "candidate_symbolic_state_id",
        "waypoint_symbolic_state_id", "metadata",
    )
    payload["candidate_payload_fingerprint"] = artifact_fingerprint(
        payload, fields
    )
    payload["nested_validity"] = {
        "passed": passed,
        "metrics": metrics,
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "dataset_fingerprint": payload["dataset_fingerprint"],
        "candidate_payload_fingerprint": payload[
            "candidate_payload_fingerprint"
        ],
        "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
        "symbolic_verifier": "igsm_rendered_operation_v1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    # A failed scientific gate is represented in the bound artifact. The
    # orchestrator decides whether to stop and can therefore write a
    # structured validity outcome instead of misclassifying this as a crash.


if __name__ == "__main__":
    main()
