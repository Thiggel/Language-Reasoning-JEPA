#!/usr/bin/env python3
"""Can the JEPA cost pick a correct next step from a fair menu?

Executability (~5%) has only ever been measured where the worker stands on its
own free-run text, which the 2026-08-13 2x2 showed is a format collapse: 0 of
4096 continuations there even parse. That confounds two very different
failures. This isolates the second one.

At a *canonical* (teacher-forced) prefix, where the frozen LM is known to emit
parseable steps, we hand the worker:

  * a fair menu  - `population` whole-sentence samples from the frozen LM, and
  * a perfect goal - the waypoint is the encoded TRUE next sentence state.

Then we ask which candidate the JEPA cost selects. Bounds are known from the
same generation call, so the number is interpretable on its own:

  floor   = pick blind        (mean validity over the menu)
  ceiling = oracle@N          (a perfect selector)
  ours    = argmin JEPA cost

Landing at the ceiling means the selection machinery works and only the format
seam is broken. Landing at the floor means the JEPA cost carries no signal
about which candidate is right, which no amount of generator fine-tuning fixes.

ORACLE / CANDIDATE-PRIVILEGED DIAGNOSTIC: the waypoint is built from the
ground-truth next step. This is not a deployable accuracy claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.data.igsm_step_verifier import (
    achieved_next_state_id,
    parse_rendered_operation,
)
from textjepa.data.provenance import sha256_file
from textjepa.planning.grounded_language_worker import build_worker_bank
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
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--k0", type=int, default=64)
    parser.add_argument("--max-roots", type=int, default=128)
    parser.add_argument(
        "--metric", choices=("euclidean", "mahalanobis"), default="euclidean",
        help="must match the metric the checkpoint was planned with",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def _encode_prefix(frozen, prefix: torch.Tensor) -> torch.Tensor:
    output = frozen(
        input_ids=prefix.unsqueeze(0),
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    return output.hidden_states[-1][0]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    records = {
        str(row["problem_id"]): row
        for row in map(json.loads, args.examples.read_text().splitlines())
    }
    tokenizer, frozen = load_reference_model(args.device, args.dtype)
    model, learner = load_hierarchical_checkpoint(args.checkpoint, args.device)
    model.eval()
    # Mirror the planner's own choice exactly (evaluate_hierarchical_language
    # _mpc._metric): euclidean runs are a plain squared distance, not the
    # learned sentence metric.
    metric = (
        learner.sentence_metric if args.metric == "mahalanobis"
        else (lambda left, right: (left - right).square().sum(-1))
    )
    dtype = next(model.parameters()).dtype

    roots = []
    for row, boundary_row in enumerate(features["boundaries"]):
        boundaries = boundary_row[boundary_row >= 0]
        roots.extend((row, step) for step in range(len(boundaries) - 1))
    order = torch.randperm(
        len(roots), generator=torch.Generator().manual_seed(args.seed)
    )[: args.max_roots]

    rows = []
    for number, root_id in enumerate(order.tolist()):
        row, step = roots[root_id]
        record = records[str(features["problem_id"][row])]
        boundaries = features["boundaries"][row]
        boundaries = boundaries[boundaries >= 0]
        # Canonical teacher-forced prefix: the true solution up to step `step`.
        prefix = features["input_ids"][row, : int(boundaries[step])].to(
            args.device
        )
        hidden = _encode_prefix(frozen, prefix).to(dtype=dtype)

        # Perfect goal: encode the true next step's endpoint as the waypoint.
        true_end = int(boundaries[step + 1])
        true_prefix = features["input_ids"][row, :true_end].to(args.device)
        true_hidden = _encode_prefix(frozen, true_prefix).to(dtype=dtype)
        # The planner (evaluate_hierarchical_language_mpc) builds its goal with
        # target=True (EMA encoders) but WorkerBank encodes candidates with the
        # ONLINE e0/e0_to_1. If that mismatch matters, selection under the
        # online waypoint will be markedly better -- which would be a planner
        # bug, not a property of the geometry. Measure both.
        waypoint = model.encode_sentence(true_hidden[-1], target=True)
        waypoint_online = model.encode_sentence(true_hidden[-1], target=False)
        # Owner's question: can we plan by descending distance to the
        # ORACLE TERMINAL state instead of a per-step waypoint? That
        # removes the manager CEM and any learned value/energy head.
        solved_end = int(features["solution_end"][row])
        solved_hidden = _encode_prefix(
            frozen, features["input_ids"][row, :solved_end].to(args.device)
        ).to(dtype=dtype)
        terminal = model.encode_sentence(solved_hidden[-1], target=True)

        bank = build_worker_bank(
            model, frozen, tokenizer, prefix, hidden,
            prompt_len=int(features["prompt_len"][row]),
            population=args.population, k0=args.k0,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            seed=args.seed * 100003 + number,
        )
        valid, parsed = [], []
        for tokens, _ in bank.candidates:
            text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
            valid.append(bool(achieved_next_state_id(text, record, step) >= 0))
            parsed.append(parse_rendered_operation(text) is not None)
        if not valid:
            continue

        # The planner's own rule: argmin over predicted-endpoint distance.
        predicted_cost = metric(bank.predicted_coarse, waypoint)
        exact_cost = metric(bank.exact_coarse, waypoint)
        chosen = int(predicted_cost.argmin())
        chosen_exact = int(exact_cost.argmin())
        chosen_online = int(metric(bank.predicted_coarse, waypoint_online).argmin())
        chosen_exact_online = int(
            metric(bank.exact_coarse, waypoint_online).argmin()
        )
        chosen_terminal = int(metric(bank.predicted_coarse, terminal).argmin())
        chosen_terminal_exact = int(
            metric(bank.exact_coarse, terminal).argmin()
        )

        rows.append({
            "problem_id": str(features["problem_id"][row]),
            "boundary_index": step,
            # The last boundary asks for "Final answer: \boxed{N}", which is a
            # different and much easier task than emitting a correct
            # operation. Kept separable so the floor/ceiling are comparable to
            # the 2026-08-13 coverage numbers.
            "is_final_boundary": step >= len(record["reasoning_operations"]),
            "n_candidates": len(valid),
            "menu_validity": sum(valid) / len(valid),      # blind-pick floor
            "menu_parse_rate": sum(parsed) / len(parsed),
            "any_valid": bool(any(valid)),                 # oracle@N ceiling
            "selected_valid": bool(valid[chosen]),         # predicted-endpoint rule
            "selected_valid_exact": bool(valid[chosen_exact]),  # exact re-encode rule
            "selected_valid_online": bool(valid[chosen_online]),
            "selected_valid_exact_online": bool(valid[chosen_exact_online]),
            "selected_valid_terminal": bool(valid[chosen_terminal]),
            "selected_valid_terminal_exact": bool(valid[chosen_terminal_exact]),
            # How close the menu can get to the requested waypoint at all.
            "min_exact_distance": float(exact_cost.min()),
            "chosen_exact_distance": float(exact_cost[chosen]),
            "valid_min_distance": (
                float(min(
                    d for d, v in zip(exact_cost.tolist(), valid) if v
                )) if any(valid) else None
            ),
        })

    def _summarize(subset: list[dict], label: str) -> dict:
        if not subset:
            return {}
        k = len(subset)
        got = [r for r in subset if r["any_valid"]]
        return {
            f"{label}roots": k,
            f"{label}floor_blind_pick": sum(
                r["menu_validity"] for r in subset
            ) / k,
            f"{label}ceiling_oracle_at_n": sum(
                r["any_valid"] for r in subset
            ) / k,
            f"{label}selected_valid_predicted": sum(
                r["selected_valid"] for r in subset
            ) / k,
            f"{label}selected_valid_exact": sum(
                r["selected_valid_exact"] for r in subset
            ) / k,
            f"{label}selected_valid_online": sum(
                r["selected_valid_online"] for r in subset
            ) / k,
            f"{label}selected_valid_exact_online": sum(
                r["selected_valid_exact_online"] for r in subset
            ) / k,
            f"{label}selected_valid_terminal": sum(
                r["selected_valid_terminal"] for r in subset
            ) / k,
            f"{label}selected_valid_terminal_exact": sum(
                r["selected_valid_terminal_exact"] for r in subset
            ) / k,
            f"{label}menu_parse_rate": sum(
                r["menu_parse_rate"] for r in subset
            ) / k,
            f"{label}mean_min_exact_distance": sum(
                r["min_exact_distance"] for r in subset
            ) / k,
            f"{label}mean_valid_min_distance": (
                sum(r["valid_min_distance"] for r in got) / len(got)
                if got else None
            ),
        }

    n = len(rows)
    scored = [r for r in rows if r["any_valid"]]
    metrics = {
        "roots": n,
        "floor_blind_pick": sum(r["menu_validity"] for r in rows) / n,
        "ceiling_oracle_at_n": sum(r["any_valid"] for r in rows) / n,
        "selected_valid_predicted": sum(r["selected_valid"] for r in rows) / n,
        "selected_valid_exact": sum(r["selected_valid_exact"] for r in rows) / n,
        "menu_parse_rate": sum(r["menu_parse_rate"] for r in rows) / n,
        # Realizability: on roots where a correct step EXISTS in the menu, does
        # the cost rank it near the top? If the best-distance candidate is
        # routinely not the valid one, the geometry is not selecting.
        "mean_min_exact_distance": sum(
            r["min_exact_distance"] for r in rows
        ) / n,
        "mean_valid_min_distance": (
            sum(r["valid_min_distance"] for r in scored) / len(scored)
            if scored else None
        ),
        # The headline split: operation boundaries are the real task.
        **_summarize(
            [r for r in rows if not r["is_final_boundary"]], "operation_"
        ),
        **_summarize(
            [r for r in rows if r["is_final_boundary"]], "final_answer_"
        ),
    }
    payload = {
        "metrics": metrics,
        "rows": rows,
        "oracle_waypoint": True,
        "candidate_privileged": True,
        "prefix_source": "canonical",
        "population": args.population,
        "k0": args.k0,
        "seed": args.seed,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset_fingerprint": features["dataset_fingerprint"],
        "backend": backend_metadata(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for key, value in metrics.items():
        print(f"{key:>28}: {value}")


if __name__ == "__main__":
    main()
