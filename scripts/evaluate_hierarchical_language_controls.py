#!/usr/bin/env python3
"""Matched greedy, flat-value, and prior-only controls for language MPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from textjepa.data.igsm_step_verifier import final_answer_matches
from textjepa.data.provenance import sha256_file
from textjepa.planning.grounded_language_worker import (
    build_worker_bank,
    encode_frozen_prefix,
    realize_macro_action,
)
from textjepa.planning.nested_language_runtime import (
    append_exact_sentence_transition,
    macro_action_log_probability,
    rollout_prior_noise,
    sentence_planning_state_from_trace,
)
from textjepa.training.hierarchical_language import ResearchStage
from textjepa.utils.language_planning_runtime import (
    backend_metadata,
    load_hierarchical_checkpoint,
    load_reference_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-split", required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--mode", choices=("greedy", "flat_value", "prior_only"), required=True)
    parser.add_argument("--metric", choices=("euclidean", "mahalanobis"), default="mahalanobis")
    parser.add_argument("--k0", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--k1", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--worker-population", type=int, default=32)
    parser.add_argument("--manager-population", type=int, default=128)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--step-cost", type=float, default=0.01)
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--worker-prior-weight", type=float, default=0.01)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--max-reasoning-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    records = {
        str(row["problem_id"]): row
        for row in map(json.loads, args.examples.read_text().splitlines())
    }
    model, learner = load_hierarchical_checkpoint(args.checkpoint, args.device)
    if args.mode == "flat_value" and (
        learner.stage != ResearchStage.VALUE_DISTILLATION or model.v is None
    ):
        raise ValueError("flat-value control requires a trained value head")
    if args.mode == "prior_only" and model.pi1 is None:
        raise ValueError("prior-only control requires Pi1")
    tokenizer, frozen = load_reference_model(args.device, args.dtype)
    metric = (
        learner.sentence_metric if args.metric == "mahalanobis"
        else lambda left, right: (left - right).square().sum(-1)
    )
    dtype = next(model.parameters()).dtype
    count = min(args.max_examples, len(features["hidden_states"]))
    order = torch.randperm(
        len(features["hidden_states"]),
        generator=torch.Generator().manual_seed(args.seed),
    )[:count]
    rows = []
    for episode, row_tensor in enumerate(order):
        row = int(row_tensor)
        record = records[str(features["problem_id"][row])]
        prompt_len = int(features["prompt_len"][row])
        prefix = features["input_ids"][row, :prompt_len].to(args.device)
        hidden = encode_frozen_prefix(frozen, prefix).to(dtype=dtype)
        boundaries = [prompt_len]
        generated = []
        timing = {name: 0.0 for name in (
            "candidate_generation_seconds", "candidate_exact_grounding_seconds",
            "token_rollout_seconds", "worker_scoring_seconds",
            "manager_seconds", "exact_reencode_seconds",
        )}
        candidate_tokens = manager_transitions = exact_tensor_tokens = 0
        failure = None
        for step in range(args.max_reasoning_steps):
            boundary_tensor = torch.tensor(boundaries, device=args.device)
            planning = sentence_planning_state_from_trace(
                model, hidden, prefix, boundary_tensor, len(boundaries) - 1
            )
            task = model.task_projection(hidden[prompt_len - 1])
            population = 1 if args.mode == "greedy" else args.worker_population
            try:
                bank = build_worker_bank(
                    model, frozen, tokenizer, prefix, hidden,
                    prompt_len=prompt_len, population=population, k0=args.k0,
                    temperature=0.8, top_p=0.95, top_k=0,
                    seed=args.seed * 1000003 + episode * 101 + step,
                )
            except RuntimeError as error:
                failure = str(error)
                break
            timing["candidate_generation_seconds"] += bank.generation_seconds
            timing["candidate_exact_grounding_seconds"] += bank.exact_grounding_seconds
            timing["token_rollout_seconds"] += bank.token_rollout_seconds
            lengths = [len(tokens) for tokens, _ in bank.candidates]
            candidate_tokens += sum(lengths)
            exact_tensor_tokens += population * (len(prefix) + max(lengths))
            scoring_started = perf_counter()
            if args.mode == "greedy":
                selected = 0
            elif args.mode == "flat_value":
                costs = []
                for index in range(population):
                    successor = append_exact_sentence_transition(
                        model, planning, bank.actions[index:index + 1],
                        bank.predicted_coarse[index:index + 1],
                    )
                    logp = macro_action_log_probability(
                        model, planning, task, bank.actions[index:index + 1]
                    )
                    costs.append(
                        args.step_cost - args.prior_weight * logp[0]
                        + model.v(successor.state, successor.context, task)[0]
                    )
                selected = int(torch.stack(costs).argmin())
            else:
                started = perf_counter()
                draws = args.manager_population * args.k1 * args.cem_iterations
                noise = torch.randn(
                    draws, 1, model.config.d_action, device=args.device,
                    dtype=dtype,
                    generator=torch.Generator(device=args.device).manual_seed(
                        args.seed * 1000003 + episode * 101 + step
                    ),
                )
                proposed = rollout_prior_noise(model, planning, task, noise)
                best = None
                for index in range(draws):
                    achieved = realize_macro_action(
                        model, planning, task, proposed.states[index, 0],
                        bank, metric,
                        worker_prior_weight=args.worker_prior_weight,
                    )
                    cost = args.step_cost - args.prior_weight * (
                        achieved.prior_log_probability[0]
                    )
                    if best is None or float(cost) < best[0]:
                        best = (float(cost), achieved.selected_index)
                assert best is not None
                selected = best[1]
                manager_transitions += draws
                timing["manager_seconds"] += perf_counter() - started
            timing["worker_scoring_seconds"] += perf_counter() - scoring_started
            selected_tokens, terminal = bank.candidates[selected]
            selected_tokens = selected_tokens.to(args.device)
            prefix = torch.cat([prefix, selected_tokens])
            generated.extend(selected_tokens.tolist())
            boundaries.append(len(prefix))
            started = perf_counter()
            hidden = encode_frozen_prefix(frozen, prefix).to(dtype=dtype)
            timing["exact_reencode_seconds"] += perf_counter() - started
            text = tokenizer.decode(selected_tokens.tolist(), skip_special_tokens=True)
            if terminal or "\\boxed" in text:
                break
        text = tokenizer.decode(generated, skip_special_tokens=True)
        total_seconds = sum(timing.values())
        rows.append({
            "problem_id": str(features["problem_id"][row]),
            "dataset_split": args.dataset_split,
            "symbolic_depth": int(record["reasoning_depth"]),
            "mode": args.mode, "metric": args.metric,
            "k0": args.k0, "k1": args.k1,
            "worker_population": population,
            "manager_population": args.manager_population,
            "cem_iterations": args.cem_iterations,
            "correct": final_answer_matches(text, int(record["answer"])),
            "generated_steps": len(boundaries) - 1,
            "generated_tokens": len(generated), "failure": failure,
            "generated_text": text, "planning_wall_seconds": total_seconds,
            "generated_candidate_tokens": candidate_tokens,
            "exact_candidate_tensor_tokens": exact_tensor_tokens,
            "manager_transition_evaluations": manager_transitions,
            **timing,
        })
    accuracy = sum(row["correct"] for row in rows) / len(rows)
    z, n = 1.959963984540054, len(rows)
    center = (accuracy + z*z/(2*n)) / (1 + z*z/n)
    half = z * ((accuracy*(1-accuracy)/n + z*z/(4*n*n)) ** 0.5) / (1+z*z/n)
    def mean(name):
        return sum(row[name] for row in rows) / len(rows)
    output = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset_fingerprint": features["dataset_fingerprint"],
        "mode": args.mode, "metric_name": args.metric,
        "accuracy": accuracy,
        "accuracy_ci95": [max(0.0, center-half), min(1.0, center+half)],
        "successful_episodes": sum(row["correct"] for row in rows),
        "episodes": len(rows),
        "generation_failure_rate": sum(row["failure"] is not None for row in rows)/len(rows),
        "mean_manager_seconds": mean("manager_seconds"),
        "mean_worker_seconds": mean("candidate_generation_seconds") + mean("candidate_exact_grounding_seconds") + mean("token_rollout_seconds") + mean("worker_scoring_seconds"),
        "mean_exact_reencode_seconds": mean("exact_reencode_seconds"),
        "mean_planning_wall_seconds": mean("planning_wall_seconds"),
        "mean_candidate_generation_seconds": mean("candidate_generation_seconds"),
        "mean_candidate_exact_grounding_seconds": mean("candidate_exact_grounding_seconds"),
        "mean_token_rollout_seconds": mean("token_rollout_seconds"),
        "mean_worker_scoring_seconds": mean("worker_scoring_seconds"),
        "mean_generated_candidate_tokens": mean("generated_candidate_tokens"),
        "mean_exact_candidate_tensor_tokens": mean("exact_candidate_tensor_tokens"),
        "mean_manager_transition_evaluations": mean("manager_transition_evaluations"),
        "backend": backend_metadata(),
        "checkpoint_metadata": {"method": args.method_label, "optimizer_step": int(torch.load(args.checkpoint, map_location="cpu", weights_only=True).get("optimizer_step", 0))},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
