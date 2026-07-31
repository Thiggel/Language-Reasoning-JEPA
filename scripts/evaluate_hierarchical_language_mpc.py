#!/usr/bin/env python3
"""Run genuine manager -> token worker -> exact re-encode iGSM MPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from textjepa.data.igsm_step_verifier import final_answer_matches
from textjepa.data.language_planning import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_VERSION,
)
from textjepa.data.provenance import sha256_file
from textjepa.planning.nested_language_runtime import (
    MacroRollout,
    append_exact_sentence_transition,
    contextual_prior_cem,
    rollout_prior_noise,
    sentence_planning_state_from_trace,
)
from textjepa.planning.grounded_language_worker import (
    build_worker_bank,
    realize_macro_action,
)
from textjepa.training.hierarchical_language import ResearchStage
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
    parser.add_argument("--replay-output", type=Path)
    parser.add_argument("--dataset-split", required=True)
    parser.add_argument("--method-label", default="hierarchical")
    parser.add_argument("--mode", choices=("oracle", "value"), required=True)
    parser.add_argument(
        "--metric", choices=("euclidean", "mahalanobis"), required=True
    )
    parser.add_argument("--k0", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--k1", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--worker-population", type=int, default=32)
    parser.add_argument("--manager-population", type=int, default=128)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--elite-fraction", type=float, default=0.1)
    parser.add_argument("--step-cost", type=float, default=0.01)
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--worker-prior-weight", type=float, default=0.01)
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--max-reasoning-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def _examples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _metric(name, learner):
    if name == "mahalanobis":
        return learner.sentence_metric
    return lambda left, right: (left - right).square().sum(-1)


@torch.no_grad()
def _encode_prefix(frozen, tokens: torch.Tensor) -> torch.Tensor:
    output = frozen(
        input_ids=tokens[None],
        attention_mask=torch.ones_like(tokens[None], dtype=torch.bool),
        output_hidden_states=True, use_cache=False, return_dict=True,
    )
    return output.hidden_states[-1][0]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if min(
        args.worker_population, args.manager_population,
        args.cem_iterations, args.max_examples, args.max_reasoning_steps,
    ) < 1:
        raise ValueError("MPC sizes must be positive")
    if min(args.step_cost, args.prior_weight, args.worker_prior_weight) < 0:
        raise ValueError("MPC costs must be nonnegative")
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    required = {
        "hidden_states", "input_ids", "prompt_len", "solution_end",
        "problem_id", "dataset_fingerprint", "model_id", "model_revision",
        "transformers_version", "symbolically_verified",
    }
    if not required <= features.keys() or not all(
        features.get("symbolically_verified", [])
    ):
        raise ValueError("MPC requires verified canonical iGSM features")
    if (
        features["model_id"] != MODEL_ID
        or features["model_revision"] != MODEL_REVISION
        or features["transformers_version"] != TRANSFORMERS_VERSION
    ):
        raise ValueError("MPC feature provenance is incompatible")
    records = {str(row["problem_id"]): row for row in _examples(args.examples)}
    model, learner = load_hierarchical_checkpoint(args.checkpoint, args.device)
    required_stage = (
        ResearchStage.MACRO_ACTION
        if args.mode == "oracle" else ResearchStage.VALUE_DISTILLATION
    )
    if learner.stage < required_stage or model.pi1 is None or (
        args.mode == "value" and model.v is None
    ):
        raise ValueError(f"{args.mode} MPC requires {required_stage.name}")
    tokenizer, frozen = load_reference_model(args.device, args.dtype)
    dtype = next(model.parameters()).dtype
    metric = _metric(args.metric, learner)
    count = min(args.max_examples, len(features["hidden_states"]))
    order = torch.randperm(
        len(features["hidden_states"]),
        generator=torch.Generator().manual_seed(args.seed),
    )[:count]
    rows = []
    planner_records = []
    grounded_transitions = []
    for episode, row_tensor in enumerate(order):
        row = int(row_tensor)
        problem_id = str(features["problem_id"][row])
        record = records[problem_id]
        prompt_len = int(features["prompt_len"][row])
        prefix = features["input_ids"][row, :prompt_len].to(args.device)
        oracle_goal = None
        if args.mode == "oracle":
            terminal_prefix = int(features["solution_end"][row])
            oracle_goal = model.encode_sentence(
                features["hidden_states"][row, terminal_prefix - 1].to(
                    args.device, dtype=dtype
                ),
                target=True,
            )
        boundaries = [prompt_len]
        generated_tokens: list[int] = []
        manager_seconds = reencode_seconds = 0.0
        generation_seconds = candidate_grounding_seconds = 0.0
        token_rollout_seconds = worker_scoring_seconds = 0.0
        generated_candidate_tokens = exact_candidate_tensor_tokens = 0
        token_transition_evaluations = manager_transition_evaluations = 0
        full_prefix_reencode_tokens = 0
        manager_diagnostics = []
        worker_exact_gaps = []
        failure = None
        started = perf_counter()
        hidden = _encode_prefix(frozen, prefix).to(dtype=dtype)
        reencode_seconds += perf_counter() - started
        full_prefix_reencode_tokens += len(prefix)
        for step in range(args.max_reasoning_steps):
            root_prefix = len(prefix)
            root_hidden_exact = hidden
            root_boundaries = list(boundaries)
            boundary_tensor = torch.tensor(
                boundaries, dtype=torch.long, device=args.device
            )
            planning_state = sentence_planning_state_from_trace(
                model, hidden, prefix, boundary_tensor,
                boundary_index=len(boundaries) - 1,
            )
            task = model.task_projection(hidden[prompt_len - 1])

            # Build one supported worker bank at the exact real prefix. The
            # same bank grounds CEM elites/random controls, so comparisons do
            # not confound manager actions with different LM proposal draws.
            try:
                bank = build_worker_bank(
                    model, frozen, tokenizer, prefix, hidden,
                    prompt_len=prompt_len, population=args.worker_population,
                    k0=args.k0, temperature=args.temperature,
                    top_p=args.top_p, top_k=args.top_k,
                    seed=args.seed * 1000003 + episode * 101 + step,
                )
            except RuntimeError as error:
                failure = str(error)
                break
            generation_seconds += bank.generation_seconds
            candidate_grounding_seconds += bank.exact_grounding_seconds
            token_rollout_seconds += bank.token_rollout_seconds
            lengths = [len(tokens) for tokens, _ in bank.candidates]
            generated_candidate_tokens += sum(lengths)
            token_transition_evaluations += sum(lengths)
            exact_candidate_tensor_tokens += len(lengths) * (
                len(prefix) + max(lengths)
            )

            def objective(rollout: MacroRollout):
                cumulative = (
                    args.step_cost - args.prior_weight
                    * rollout.log_probabilities
                ).cumsum(-1)
                if args.mode == "oracle":
                    assert oracle_goal is not None
                    continuation = metric(
                        rollout.states, oracle_goal
                    )
                else:
                    continuation = model.v(
                        rollout.states, rollout.contexts, task
                    )
                return (cumulative + continuation).min(-1)

            def ground_manager(noise, predicted_rollout, ids):
                nonlocal worker_scoring_seconds
                grounded_costs = []
                for local, manager_id in enumerate(ids.tolist()):
                    waypoint_i = predicted_rollout.states[manager_id, 0]
                    score_started = perf_counter()
                    achieved = realize_macro_action(
                        model, planning_state, task, waypoint_i, bank, metric,
                        worker_prior_weight=args.worker_prior_weight,
                    )
                    worker_scoring_seconds += perf_counter() - score_started
                    states_i = achieved.planning_state.state[:, None]
                    contexts_i = achieved.planning_state.context[:, None]
                    logp_i = achieved.prior_log_probability[:, None]
                    if args.k1 > 1:
                        continuation_rollout = rollout_prior_noise(
                            model, achieved.planning_state, task,
                            noise[local:local + 1, 1:],
                        )
                        states_i = torch.cat([
                            states_i, continuation_rollout.states
                        ], 1)
                        contexts_i = torch.cat([
                            contexts_i, continuation_rollout.contexts
                        ], 1)
                        logp_i = torch.cat([
                            logp_i,
                            continuation_rollout.log_probabilities,
                        ], 1)
                    cumulative_i = (
                        args.step_cost - args.prior_weight * logp_i
                    ).cumsum(-1)
                    continuation_i = (
                        metric(states_i, oracle_goal)
                        if args.mode == "oracle"
                        else model.v(states_i, contexts_i, task)
                    )
                    grounded_costs.append(
                        (cumulative_i + continuation_i).min()
                    )
                return torch.stack(grounded_costs)

            started = perf_counter()
            manager = contextual_prior_cem(
                model, planning_state, task, objective,
                horizon=args.k1, population=args.manager_population,
                iterations=args.cem_iterations,
                elite_fraction=args.elite_fraction,
                ground=ground_manager,
                ground_topn=max(
                    1, round(args.manager_population * args.elite_fraction)
                ),
                ground_random=max(
                    1, round(args.manager_population * args.elite_fraction)
                ),
                select_grounded=True,
                generator=torch.Generator(device=args.device).manual_seed(
                    args.seed * 1000003 + episode * 101 + step
                ),
            )
            manager_seconds += perf_counter() - started
            manager_diagnostics.extend(manager.diagnostics)
            manager_transition_evaluations += (
                args.manager_population * args.k1 * args.cem_iterations
                + args.k1
                + sum(
                    int(row["grounded_candidates"]) * max(args.k1 - 1, 0)
                    for row in manager.diagnostics
                )
            )
            waypoint = manager.rollout.states[0, 0]
            predicted_cost = metric(bank.predicted_coarse, waypoint) - (
                args.worker_prior_weight * bank.lm_log_probability
            )
            exact_cost = metric(bank.exact_coarse, waypoint) - (
                args.worker_prior_weight * bank.lm_log_probability
            )
            selected = int(predicted_cost.argmin())
            worker_exact_gaps.append(float(
                exact_cost[selected] - exact_cost.min()
            ))
            selected_tokens, selected_terminal = bank.candidates[selected]
            selected_tokens = selected_tokens.to(args.device)
            prefix = torch.cat([prefix, selected_tokens])
            generated_tokens.extend(selected_tokens.tolist())
            boundaries.append(len(prefix))
            # Hierarchical MPC always discards imagined endpoints and rebuilds
            # the exact frozen-LM state/cache after executing real text.
            started = perf_counter()
            hidden = _encode_prefix(frozen, prefix).to(dtype=dtype)
            reencode_seconds += perf_counter() - started
            full_prefix_reencode_tokens += len(prefix)
            if args.replay_output is not None:
                token_begin = max(
                    prompt_len, root_prefix - model.config.token_context + 1
                )
                token_prefix_lengths = torch.arange(
                    token_begin, root_prefix + 1, device=args.device
                )
                sentence_begin = max(
                    0, len(root_boundaries) - model.config.sentence_context
                )
                sentence_positions = root_boundaries[sentence_begin:]
                planner_records.append({
                    "source": "sample_t0.8",
                    "temperature": args.temperature,
                    "token_ids": selected_tokens.detach().cpu(),
                    "root_hidden": root_hidden_exact[root_prefix - 1].detach().cpu(),
                    "suffix_hidden": hidden[root_prefix:].detach().cpu(),
                    "sentence_eligible": True,
                    "completed_boundary": True,
                    "terminal_eos": bool(selected_terminal),
                    "lm_log_probability": bank.lm_log_probability[selected].detach().cpu(),
                    "root_token_history_hidden": root_hidden_exact[
                        token_prefix_lengths - 1
                    ].detach().cpu(),
                    "root_token_history_action_ids": prefix[
                        token_begin:root_prefix
                    ].detach().cpu(),
                    "root_sentence_history_hidden": root_hidden_exact[
                        torch.tensor(sentence_positions, device=args.device) - 1
                    ].detach().cpu(),
                    "root_sentence_history_spans": [
                        prefix[root_boundaries[index]:root_boundaries[index + 1]].detach().cpu()
                        for index in range(sentence_begin, len(root_boundaries) - 1)
                    ],
                })
                grounded_transitions.append({
                    "candidate_source": "worker_achieved",
                    "cem_iteration": max(args.cem_iterations - 1, 0),
                    "exact_reencoded": True,
                    "worker_achieved": True,
                    "predicted_endpoint": bank.predicted_coarse[selected].detach().cpu(),
                    "achieved_endpoint": bank.exact_coarse[selected].detach().cpu(),
                    "problem_id": problem_id,
                    "root_prefix_length": root_prefix,
                })
            decoded_step = tokenizer.decode(
                selected_tokens.tolist(), skip_special_tokens=True
            )
            if selected_terminal or "\\boxed" in decoded_step:
                break
        generated_text = tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        correct = final_answer_matches(generated_text, int(record["answer"]))
        rows.append({
            "problem_id": problem_id,
            "dataset_split": args.dataset_split,
            "symbolic_depth": int(record["reasoning_depth"]),
            "mode": args.mode,
            "metric": args.metric,
            "k0": args.k0,
            "k1": args.k1,
            "worker_population": args.worker_population,
            "manager_population": args.manager_population,
            "cem_iterations": args.cem_iterations,
            "correct": bool(correct),
            "generated_steps": len(boundaries) - 1,
            "generated_tokens": len(generated_tokens),
            "mpc_replans": len(boundaries) - 1,
            "generated_candidate_tokens": generated_candidate_tokens,
            "exact_candidate_tensor_tokens": exact_candidate_tensor_tokens,
            "token_transition_evaluations": token_transition_evaluations,
            "manager_transition_evaluations": manager_transition_evaluations,
            "full_prefix_reencode_tokens": full_prefix_reencode_tokens,
            "manager_seconds": manager_seconds,
            "candidate_generation_seconds": generation_seconds,
            "candidate_exact_grounding_seconds": candidate_grounding_seconds,
            "token_rollout_seconds": token_rollout_seconds,
            "worker_scoring_seconds": worker_scoring_seconds,
            "worker_seconds": (
                generation_seconds + candidate_grounding_seconds
                + token_rollout_seconds + worker_scoring_seconds
            ),
            "planning_wall_seconds": (
                manager_seconds + generation_seconds
                + candidate_grounding_seconds + token_rollout_seconds
                + worker_scoring_seconds + reencode_seconds
            ),
            "exact_reencode_seconds": reencode_seconds,
            "mean_worker_exact_gap": (
                sum(worker_exact_gaps) / len(worker_exact_gaps)
                if worker_exact_gaps else None
            ),
            "failure": failure,
            "generated_text": generated_text,
            "manager_diagnostics": manager_diagnostics,
        })
        print(json.dumps({
            key: value for key, value in rows[-1].items()
            if key not in {"generated_text", "manager_diagnostics"}
        }, sort_keys=True), flush=True)
    accuracy = sum(row["correct"] for row in rows) / len(rows)
    # Wilson interval remains meaningful for small pilot cells and does not
    # pretend a single-seed estimate is asymptotically Gaussian.
    z = 1.959963984540054
    n = len(rows)
    center = (accuracy + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) ** 0.5) / (
        1 + z * z / n
    )
    output = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset_fingerprint": features["dataset_fingerprint"],
        "mode": args.mode,
        "metric_name": args.metric,
        "accuracy": accuracy,
        "accuracy_ci95": [max(0.0, center - half), min(1.0, center + half)],
        "successful_episodes": sum(row["correct"] for row in rows),
        "episodes": len(rows),
        "generation_failure_rate": sum(
            row["failure"] is not None for row in rows
        ) / len(rows),
        "mean_manager_seconds": sum(
            row["manager_seconds"] for row in rows
        ) / len(rows),
        "mean_worker_seconds": sum(
            row["worker_seconds"] for row in rows
        ) / len(rows),
        "mean_candidate_generation_seconds": sum(
            row["candidate_generation_seconds"] for row in rows
        ) / len(rows),
        "mean_candidate_exact_grounding_seconds": sum(
            row["candidate_exact_grounding_seconds"] for row in rows
        ) / len(rows),
        "mean_token_rollout_seconds": sum(
            row["token_rollout_seconds"] for row in rows
        ) / len(rows),
        "mean_worker_scoring_seconds": sum(
            row["worker_scoring_seconds"] for row in rows
        ) / len(rows),
        "mean_exact_reencode_seconds": sum(
            row["exact_reencode_seconds"] for row in rows
        ) / len(rows),
        "mean_planning_wall_seconds": sum(
            row["planning_wall_seconds"] for row in rows
        ) / len(rows),
        "mean_generated_candidate_tokens": sum(
            row["generated_candidate_tokens"] for row in rows
        ) / len(rows),
        "mean_exact_candidate_tensor_tokens": sum(
            row["exact_candidate_tensor_tokens"] for row in rows
        ) / len(rows),
        "mean_token_transition_evaluations": sum(
            row["token_transition_evaluations"] for row in rows
        ) / len(rows),
        "mean_manager_transition_evaluations": sum(
            row["manager_transition_evaluations"] for row in rows
        ) / len(rows),
        "mean_full_prefix_reencode_tokens": sum(
            row["full_prefix_reencode_tokens"] for row in rows
        ) / len(rows),
        "mean_optimizer_curse_regret": (
            sum(
                diagnostic["optimizer_curse_regret"]
                for row in rows
                for diagnostic in row["manager_diagnostics"]
                if diagnostic["optimizer_curse_regret"]
                == diagnostic["optimizer_curse_regret"]
            )
            / max(1, sum(
                diagnostic["optimizer_curse_regret"]
                == diagnostic["optimizer_curse_regret"]
                for row in rows
                for diagnostic in row["manager_diagnostics"]
            ))
        ),
        "backend": backend_metadata(),
        "checkpoint_metadata": {
            "method": args.method_label,
            "optimizer_step": int(torch.load(
                args.checkpoint, map_location="cpu", weights_only=True
            ).get("optimizer_step", 0)),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    if args.replay_output is not None:
        replay = {
            "architecture": torch.load(
                args.checkpoint, map_location="cpu", weights_only=True
            )["architecture"],
            "model_revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "dataset_fingerprint": features["dataset_fingerprint"],
            "source_checkpoint_sha256": sha256_file(args.checkpoint),
            "exact_reencoded": True,
            "grounded_transitions": grounded_transitions,
            "counterfactual_records": planner_records,
        }
        args.replay_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(replay, args.replay_output)


if __name__ == "__main__":
    main()
