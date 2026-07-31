#!/usr/bin/env python3
"""Materialize contextual macro rollouts toward verified iGSM terminals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from textjepa.data.language_planning import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_VERSION,
)
from textjepa.data.provenance import artifact_fingerprint, sha256_file
from textjepa.data.igsm_step_verifier import achieved_next_state_id
from textjepa.models.hierarchical_language_jepa import (
    HIERARCHICAL_LANGUAGE_ARCHITECTURE,
)
from textjepa.planning.nested_language_runtime import (
    advance_sentence_planning_state,
    rollout_prior_noise,
    sentence_planning_state_from_trace,
)
from textjepa.planning.grounded_language_worker import (
    build_worker_bank,
    encode_frozen_prefix,
    realize_macro_action,
)
from textjepa.training.hierarchical_language import ResearchStage
from textjepa.utils.language_planning_runtime import (
    load_hierarchical_checkpoint,
    load_reference_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-roots", type=int, default=256)
    parser.add_argument("--first-actions", type=int, default=16)
    parser.add_argument("--continuation-samples", type=int, default=4)
    parser.add_argument("--max-prefix", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker-population", type=int, default=32)
    parser.add_argument("--k0", type=int, default=64)
    parser.add_argument("--worker-prior-weight", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--latent-diagnostic", action="store_true",
        help="Build ungrounded P1-only rollouts; rejected by value training.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _examples(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    result = {str(row["problem_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate problem IDs in oracle-rollout examples")
    return result


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
def _build_grounded_payload(args, features, model, learner, roots):
    if args.examples is None:
        raise ValueError("grounded oracle rollouts require --examples")
    if min(args.worker_population, args.k0) < 1 or (
        args.worker_prior_weight < 0
    ):
        raise ValueError("invalid worker-grounding configuration")
    records = _examples(args.examples)
    tokenizer, frozen = load_reference_model(args.device, args.dtype)
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 91)
    first, samples, depth = (
        args.first_actions, args.continuation_samples, args.max_prefix
    )
    all_states, all_logp, all_masks, all_symbolic = [], [], [], []
    all_actions, all_requested, all_displacement = [], [], []
    successors, successor_contexts, task_hidden_rows, goals = [], [], [], []
    metadata = []
    timing = {
        "candidate_generation_seconds": 0.0,
        "candidate_exact_grounding_seconds": 0.0,
        "token_rollout_seconds": 0.0,
        "full_prefix_reencode_seconds": 0.0,
    }
    metric = learner.sentence_metric
    for root_number, (row, boundary_index) in enumerate(roots):
        problem_id = str(features["problem_id"][row])
        if problem_id not in records:
            raise ValueError("feature/example problem IDs do not align")
        record = records[problem_id]
        boundaries = _valid_boundaries(features["boundaries"][row]).to(
            args.device
        )
        hidden = features["hidden_states"][row].to(args.device, dtype=dtype)
        ids = features["input_ids"][row].to(args.device)
        prompt_len = int(features["prompt_len"][row])
        root_prefix = int(boundaries[boundary_index])
        prefix0 = ids[:root_prefix]
        planning0 = sentence_planning_state_from_trace(
            model, hidden, ids, boundaries, boundary_index
        )
        task_hidden = hidden[prompt_len - 1]
        task = model.task_projection(task_hidden)
        first_noise = torch.randn(
            first, model.config.d_action, device=args.device, dtype=dtype,
            generator=generator,
        )
        continuation_noise = torch.randn(
            first, samples, max(depth - 1, 0), model.config.d_action,
            device=args.device, dtype=dtype, generator=generator,
        )
        root_states = torch.zeros(
            first, samples, depth, model.config.d_sentence, dtype=dtype,
            device=args.device,
        )
        root_logp = torch.zeros(
            first, samples, depth, dtype=torch.float32, device=args.device
        )
        root_mask = torch.zeros(
            first, samples, depth, dtype=torch.bool, device=args.device
        )
        root_symbolic = torch.zeros_like(root_mask)
        root_actions = torch.zeros(
            first, samples, depth, model.config.d_action, dtype=dtype,
            device=args.device,
        )
        root_requested = torch.zeros_like(root_actions)
        root_displacement = torch.zeros(
            first, samples, depth, dtype=torch.float32, device=args.device
        )
        first_successors, first_contexts, first_logps = [], [], []
        # All candidate first actions share the same exact textual root. Build
        # one frozen-LM worker bank and score every requested waypoint against
        # it; regenerating an identical bank per latent action is wasteful and
        # confounds action ranking with proposal noise.
        root_bank = build_worker_bank(
            model, frozen, tokenizer, prefix0, hidden,
            prompt_len=prompt_len, population=args.worker_population,
            k0=args.k0, temperature=args.temperature, top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed * 1000003 + root_number * 1009,
        )
        timing["candidate_generation_seconds"] += root_bank.generation_seconds
        timing["candidate_exact_grounding_seconds"] += (
            root_bank.exact_grounding_seconds
        )
        timing["token_rollout_seconds"] += root_bank.token_rollout_seconds
        for action_index in range(first):
            mean, logvar = model.pi1.prior_params(
                planning0.state, planning0.context, task[None]
            )
            requested = mean + (0.5 * logvar).exp() * first_noise[
                action_index:action_index + 1
            ]
            predicted = advance_sentence_planning_state(
                model, planning0, requested
            )
            achieved = realize_macro_action(
                model, planning0, task, predicted.state[0], root_bank, metric,
                worker_prior_weight=args.worker_prior_weight,
            )
            first_successors.append(achieved.planning_state.state[0])
            first_contexts.append(achieved.planning_state.context[0])
            first_logps.append(achieved.prior_log_probability[0])
            prefix1 = torch.cat([prefix0, achieved.tokens])
            started = perf_counter()
            hidden1 = encode_frozen_prefix(frozen, prefix1).to(dtype=dtype)
            timing["full_prefix_reencode_seconds"] += perf_counter() - started
            text = tokenizer.decode(
                achieved.tokens.tolist(), skip_special_tokens=True
            )
            symbolic_ok = achieved_next_state_id(
                text, record, boundary_index
            ) >= 0
            for sample_index in range(samples):
                state = achieved.planning_state
                prefix = prefix1
                current_hidden = hidden1
                valid_path = symbolic_ok
                root_states[action_index, sample_index, 0] = state.state[0]
                root_logp[action_index, sample_index, 0] = (
                    achieved.prior_log_probability[0]
                )
                root_mask[action_index, sample_index, 0] = True
                root_symbolic[action_index, sample_index, 0] = valid_path
                root_actions[action_index, sample_index, 0] = achieved.action[0]
                root_requested[action_index, sample_index, 0] = requested[0]
                root_displacement[action_index, sample_index, 0] = (
                    achieved.action[0] - requested[0]
                ).float().norm()
                terminal = achieved.terminal
                for step in range(1, depth):
                    if terminal:
                        break
                    mean, logvar = model.pi1.prior_params(
                        state.state, state.context, task[None]
                    )
                    requested = mean + (0.5 * logvar).exp() * (
                        continuation_noise[
                            action_index, sample_index, step - 1
                        ][None]
                    )
                    predicted = advance_sentence_planning_state(
                        model, state, requested
                    )
                    bank = build_worker_bank(
                        model, frozen, tokenizer, prefix, current_hidden,
                        prompt_len=prompt_len,
                        population=args.worker_population, k0=args.k0,
                        temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k,
                        seed=(args.seed * 1000003 + root_number * 1009
                              + action_index * 97 + sample_index * 17 + step),
                    )
                    timing["candidate_generation_seconds"] += (
                        bank.generation_seconds
                    )
                    timing["candidate_exact_grounding_seconds"] += (
                        bank.exact_grounding_seconds
                    )
                    timing["token_rollout_seconds"] += bank.token_rollout_seconds
                    achieved = realize_macro_action(
                        model, state, task, predicted.state[0], bank, metric,
                        worker_prior_weight=args.worker_prior_weight,
                    )
                    state = achieved.planning_state
                    prefix = torch.cat([prefix, achieved.tokens])
                    started = perf_counter()
                    current_hidden = encode_frozen_prefix(
                        frozen, prefix
                    ).to(dtype=dtype)
                    timing["full_prefix_reencode_seconds"] += (
                        perf_counter() - started
                    )
                    text = tokenizer.decode(
                        achieved.tokens.tolist(), skip_special_tokens=True
                    )
                    step_ok = False
                    if valid_path and boundary_index + step < len(
                        record["canonical_state_ids"]
                    ) - 1:
                        step_ok = achieved_next_state_id(
                            text, record, boundary_index + step
                        ) >= 0
                    valid_path = valid_path and step_ok
                    root_states[action_index, sample_index, step] = state.state[0]
                    root_logp[action_index, sample_index, step] = (
                        achieved.prior_log_probability[0]
                    )
                    root_mask[action_index, sample_index, step] = True
                    root_symbolic[action_index, sample_index, step] = valid_path
                    root_actions[action_index, sample_index, step] = (
                        achieved.action[0]
                    )
                    root_requested[action_index, sample_index, step] = requested[0]
                    root_displacement[action_index, sample_index, step] = (
                        achieved.action[0] - requested[0]
                    ).float().norm()
                    terminal = achieved.terminal
        all_states.append(root_states.cpu())
        all_logp.append(root_logp.cpu())
        all_masks.append(root_mask.cpu())
        all_symbolic.append(root_symbolic.cpu())
        all_actions.append(root_actions.cpu())
        all_requested.append(root_requested.cpu())
        all_displacement.append(root_displacement.cpu())
        successors.append(torch.stack(first_successors).cpu())
        successor_contexts.append(torch.stack(first_contexts).cpu())
        task_hidden_rows.append(task_hidden.cpu())
        terminal_prefix = int(features["solution_end"][row])
        terminal = model.encode_sentence(
            hidden[terminal_prefix - 1], target=True
        )
        goals.append(terminal[None].cpu())
        metadata.append({
            "problem_id": problem_id,
            "root_boundary_index": boundary_index,
            "root_prefix_length": root_prefix,
            "symbolic_depth": int(features["reasoning_depth"][row]),
        })
    states = torch.stack(all_states)
    logp = torch.stack(all_logp)
    mask = torch.stack(all_masks)
    payload = {
        "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_feature_sha256": sha256_file(args.features),
        "source_examples_sha256": sha256_file(args.examples),
        "dataset_fingerprint": features["dataset_fingerprint"],
        "rollout_states": states,
        "rollout_log_probabilities": logp,
        "rollout_mask": mask,
        "rollout_symbolically_verified": torch.stack(all_symbolic),
        "achieved_actions": torch.stack(all_actions),
        "requested_actions": torch.stack(all_requested),
        "requested_achieved_displacement": torch.stack(all_displacement),
        "goals": torch.stack(goals),
        "goal_mask": torch.ones(len(goals), 1, dtype=torch.bool),
        "successor_state": torch.stack(successors),
        "successor_context": torch.stack(successor_contexts),
        "task_hidden": torch.stack(task_hidden_rows),
        "first_action_log_probability": logp[:, :, 0, 0],
        "action_mask": mask[:, :, 0, 0],
        "metadata": metadata,
        "terminal_set_symbolically_verified": True,
        "rollouts_exactly_grounded": True,
        "timing": timing,
        "sampling": {
            "first_actions": first, "continuation_samples": samples,
            "total_prefixes": depth, "seed": args.seed,
            "prior_coordinate_sampling": True,
            "context_history_preserved": True,
            "worker_population": args.worker_population, "k0": args.k0,
            "worker_prior_weight": args.worker_prior_weight,
        },
    }
    payload["terminal_set_fingerprint"] = artifact_fingerprint(
        payload, ("goals", "goal_mask", "metadata")
    )
    payload["oracle_rollout_fingerprint"] = artifact_fingerprint(payload, (
        "rollout_states", "rollout_log_probabilities", "rollout_mask",
        "rollout_symbolically_verified", "achieved_actions",
        "requested_actions", "requested_achieved_displacement",
        "successor_state", "successor_context", "task_hidden",
        "first_action_log_probability", "action_mask", "metadata",
        "checkpoint_sha256", "dataset_fingerprint",
    ))
    return payload


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
    if not args.latent_diagnostic:
        payload = _build_grounded_payload(
            args, features, model, learner, roots
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.output)
        return
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
        "terminal_set_symbolically_verified": True,
        "rollouts_exactly_grounded": False,
        "sampling": {
            "first_actions": first,
            "continuation_samples": samples,
            "total_prefixes": depth,
            "seed": args.seed,
            "prior_coordinate_sampling": True,
            "context_history_preserved": True,
            "diagnostic_only": True,
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
