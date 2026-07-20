#!/usr/bin/env python3
"""Evaluate nested subgoal CEM and macro-option decoder planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from textjepa.data.faithful_token_edits import FaithfulTokenEditDataset
from textjepa.data.token_edit_distance import boundary_token_edit_distance
from textjepa.planning.hierarchical_edit_cem import HierarchicalEditCEM
from textjepa.planning.multiscale_edit_mpc import MultiscaleEditMPC, copy_buffer, flatten
from textjepa.utils.checkpoint import load_run


REGIMES = {
    "id": ((8, 21), 21, 28, 7401),
    "ood_medium": ((22, 24), 28, 36, 7402),
    "ood_long": ((25, 28), 28, 36, 7403),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=HierarchicalEditCEM.MODES, required=True)
    parser.add_argument("--regime", choices=REGIMES, default="id")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--high-horizon", type=int, default=1)
    parser.add_argument("--low-horizon", type=int, default=4)
    parser.add_argument("--cem-candidates", type=int, default=32)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--cem-elites", type=int, default=4)
    parser.add_argument("--reachability-topk", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--top-positions", type=int, default=4)
    parser.add_argument("--top-tokens", type=int, default=4)
    args = parser.parse_args()

    model, vocab, _ = load_run(args.ckpt, args.device)
    op_range, max_op, max_edge, seed = REGIMES[args.regime]
    dataset = FaithfulTokenEditDataset(
        vocab, size=args.examples, seed=seed, max_op=max_op,
        max_edge=max_edge, op_range=op_range,
        corruption_mode="iterative_refinement", trajectory_variants=1,
        refinement_probability=0.0, min_edits=1, max_edits=1,
    )
    primitive = MultiscaleEditMPC(
        model, vocab, args.device, beam_width=args.beam_width,
        top_positions=args.top_positions, top_tokens=args.top_tokens,
        max_candidates=args.top_positions * args.top_tokens,
    )
    planner = HierarchicalEditCEM(
        primitive, mode=args.mode, high_horizon=args.high_horizon,
        candidates=args.cem_candidates, iterations=args.cem_iterations,
        elites=args.cem_elites, reachability_topk=args.reachability_topk,
        low_horizon=args.low_horizon,
    )
    episodes = []
    for index in range(args.examples):
        item = dataset[index]
        current = copy_buffer(item["buffers"][0])
        target = copy_buffer(item["buffers"][-1])
        initial = boundary_token_edit_distance(current, target)
        budget = args.max_actions if args.max_actions > 0 else len(flatten(current))
        residuals = []
        decoded = []
        for _ in range(budget):
            plan = planner.first_action(item["prompt"], current)
            if plan.first_action is None:
                break
            from textjepa.data.faithful_token_edits import _apply
            _apply(current, plan.first_action)
            residuals.append(plan.reachability_residual)
            decoded.append(len(plan.decoded_actions))
            if current == target:
                break
        final = boundary_token_edit_distance(current, target)
        episode = {
            "initial_distance": initial,
            "final_distance": final,
            "normalized_improvement": (initial - final) / max(initial, 1),
            "token_accuracy": sum(
                int(a == b) for a, b in zip(flatten(current), flatten(target))
            ) / max(len(flatten(target)), 1),
            "exact_sequence": current == target,
            "exact_answer_sentence": current[-1] == target[-1],
            "actions": len(residuals),
            "mean_reachability_residual": (
                sum(residuals) / max(len(residuals), 1)
            ),
            "mean_decoded_option_length": sum(decoded) / max(len(decoded), 1),
        }
        if index < 2:
            episode["generated_text"] = [vocab.decode(x) for x in current]
            episode["target_text"] = [vocab.decode(x) for x in target]
        episodes.append(episode)
    mean = lambda key: sum(float(x[key]) for x in episodes) / len(episodes)
    payload = {
        "information_regime": (
            "target-free nested planning; clean target is evaluator-only; "
            "known sentence and token slots"
        ),
        "mode": args.mode,
        "regime": args.regime,
        "settings": {k: v for k, v in vars(args).items()
                     if k not in {"ckpt", "out", "device"}},
        "normalized_edit_distance_improvement": mean("normalized_improvement"),
        "token_accuracy": mean("token_accuracy"),
        "exact_sequence_rate": mean("exact_sequence"),
        "exact_answer_sentence_rate": mean("exact_answer_sentence"),
        "mean_reachability_residual": mean("mean_reachability_residual"),
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()

