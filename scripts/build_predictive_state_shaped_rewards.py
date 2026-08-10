#!/usr/bin/env python3
"""Build frozen-distance potential rewards for a terminal-reward RL run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.data.predictive_state import validate_reasoning_bundle
from textjepa.planning.predictive_state import (
    make_goal_distance_model,
    potential_shaped_reward,
)


def calibrate_potential_weight(
    distance_deltas: list[torch.Tensor],
    requested_weight: float,
    maximum_absolute_return: float = 0.5,
) -> float:
    """Choose one fixed coefficient for every trajectory in a dataset."""
    if requested_weight < 0 or maximum_absolute_return <= 0:
        raise ValueError("invalid shaping calibration")
    maximum = max(
        (float(delta.abs().sum()) for delta in distance_deltas),
        default=0.0,
    )
    if maximum == 0:
        return requested_weight
    return min(requested_weight, maximum_absolute_return / maximum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--goal-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--weight", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    validate_reasoning_bundle(features)
    checkpoint = torch.load(
        args.goal_checkpoint, map_location="cpu", weights_only=True
    )
    model = make_goal_distance_model(
        checkpoint.get("distance_kind", "euclidean"),
        int(checkpoint["state_size"]), int(checkpoint["geometry_size"]),
    ).to(args.device)
    model.load_state_dict(checkpoint["distance_model"])
    model.eval()
    prepared = []
    with torch.no_grad():
        for record in features["records"]:
            valid = record["valid"].bool()
            states = record["states"][valid].float().to(args.device)
            prompt = record["prompt_state"].float().to(args.device)
            distance = model(states, prompt)
            if len(distance) < 2:
                continue
            delta = distance[:-1] - args.gamma * distance[1:]
            prepared.append((record, distance, delta))
    effective_weight = calibrate_potential_weight(
        [row[2] for row in prepared], args.weight, 0.5
    )
    records, totals = [], []
    with torch.no_grad():
        for record, distance, _delta in prepared:
            terminal = torch.zeros(len(distance) - 1, device=args.device)
            if bool(record["correct"]):
                terminal[-1] = 1.0
            shaped, increment = potential_shaped_reward(
                terminal, distance[:-1], distance[1:], gamma=args.gamma,
                weight=effective_weight, clip=args.clip,
            )
            totals.append(float(increment.abs().sum()))
            records.append({
                "problem_id": str(record["problem_id"]),
                "trajectory_id": str(record["trajectory_id"]),
                "terminal_rewards": terminal.cpu(),
                "distance": distance.cpu(),
                "shaping_increment": increment.cpu(),
                "shaped_rewards": shaped.cpu(),
            })
    output = {
        "schema_version": 1,
        "kind": "frozen_predictive_state_potential_rewards",
        "metadata": {
            "features": str(args.features),
            "goal_checkpoint": str(args.goal_checkpoint),
            "gamma": args.gamma,
            "requested_weight": args.weight,
            "effective_fixed_weight": effective_weight,
            "increment_clip": args.clip,
            "maximum_absolute_shaping_return": 0.5,
            "weight_calibration_scope": "single_global_dataset_coefficient",
            "distance_model_frozen": True,
            "exact_terminal_verifier_retained": True,
            "policy_invariance_guarantee_claimed": False,
            "candidate_privileged_verified_outcomes": True,
        },
        "mean_absolute_shaping_return": (
            sum(totals) / len(totals) if totals else 0.0
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps(output["metadata"] | {
        "records": len(records),
        "mean_absolute_shaping_return": output["mean_absolute_shaping_return"],
    }), flush=True)


if __name__ == "__main__":
    main()
