#!/usr/bin/env python3
"""Stage 3A oracle terminal-state geometry probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from textjepa.analysis.predictive_state import (
    candidate_ranking_accuracy,
    progress_metrics,
)
from textjepa.data.predictive_state import validate_reasoning_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--mahalanobis-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def training_problem_ids(records: list[dict]) -> set[str]:
    problems = sorted(
        {str(record["problem_id"]) for record in records},
        key=lambda value: hashlib.sha256(value.encode()).digest(),
    )
    if len(problems) < 2:
        raise ValueError("oracle geometry needs at least two problems")
    test_count = max(1, round(0.2 * len(problems)))
    return set(problems[test_count:])


def fit_transforms(records: list[dict], pca_dim: int, train_ids: set[str]) -> dict:
    states = torch.cat([
        record["states"][record["valid"].bool()].float()
        for record in records if str(record["problem_id"]) in train_ids
    ])
    mean = states.mean(0)
    centered = states - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    dimension = min(pca_dim, states.shape[-1], len(states) - 1)
    return {
        "mean": mean,
        "pca": eigenvectors[:, :dimension],
        "whitening": eigenvectors[:, :dimension]
        / eigenvalues[:dimension].clamp_min(1e-6).sqrt(),
    }


def fit_mahalanobis(records: list[dict], width: int, dimension: int,
                    steps: int, seed: int, train_ids: set[str]) -> torch.Tensor:
    torch.manual_seed(seed)
    projection = torch.nn.Linear(width, dimension, bias=False)
    torch.nn.init.orthogonal_(projection.weight)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=1e-3,
                                  weight_decay=1e-3)
    examples = []
    for record in records:
        if not record["correct"] or str(record["problem_id"]) not in train_ids:
            continue
        valid = record["valid"].bool()
        state = record["states"][valid].float()
        terminal = record["states"][int(record["terminal_index"])].float()
        remaining = record["remaining_chunks"][valid].float()
        examples.append((state, terminal, remaining))
    if not examples:
        raise ValueError("no correct training trajectories for Mahalanobis fit")
    for step in range(steps):
        state, terminal, remaining = examples[step % len(examples)]
        distance = (projection(state) - projection(terminal)).norm(dim=-1)
        loss = torch.nn.functional.huber_loss(distance, remaining)
        loss = loss + 0.1 * distance[-1].square()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return projection.weight.detach().T


def distances(record: dict, transform: torch.Tensor | None,
              kind: str, mean: torch.Tensor | None = None) -> torch.Tensor:
    valid = record["valid"].bool()
    state = record["states"][valid].float()
    goal = record["states"][int(record["terminal_index"])].float()
    if kind == "cosine":
        return 1.0 - torch.nn.functional.cosine_similarity(
            state, goal[None], dim=-1
        )
    if mean is not None:
        state = state - mean
        goal = goal - mean
    if transform is not None:
        state = state @ transform
        goal = goal @ transform
    return (state - goal).norm(dim=-1)


def remaining_r2(train_rows, test_rows) -> float:
    if not train_rows or not test_rows:
        return float("nan")
    x_train = torch.cat([row[0] for row in train_rows])
    y_train = torch.cat([row[1] for row in train_rows])
    design = torch.stack([x_train, torch.ones_like(x_train)], dim=-1)
    coefficients = torch.linalg.lstsq(design, y_train[:, None]).solution[:, 0]
    x_test = torch.cat([row[0] for row in test_rows])
    y_test = torch.cat([row[1] for row in test_rows])
    prediction = coefficients[0] * x_test + coefficients[1]
    residual = (y_test - prediction).square().sum()
    total = (y_test - y_test.mean()).square().sum()
    return float(1.0 - residual / total.clamp_min(1e-12))


def main() -> None:
    args = parse_args()
    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    validate_reasoning_bundle(payload)
    records = payload["records"]
    train_ids = training_problem_ids(records)
    width = records[0]["states"].shape[-1]
    fitted = fit_transforms(records, args.pca_dim, train_ids)
    mahalanobis = fit_mahalanobis(
        records, width, min(args.pca_dim, width),
        args.mahalanobis_steps, args.seed, train_ids,
    )
    variants = {
        "cosine": (None, None),
        "euclidean": (None, None),
        "whitened_euclidean": (fitted["whitening"], fitted["mean"]),
        "pca_euclidean": (fitted["pca"], fitted["mean"]),
        "learned_linear_mahalanobis": (mahalanobis, None),
    }
    results = {}
    for name, (transform, mean) in variants.items():
        train_rows, test_rows, test_distances = [], [], []
        by_problem: dict[str, list[tuple[dict, torch.Tensor]]] = {}
        for record in records:
            if not record["correct"]:
                continue
            distance = distances(
                record, transform, "cosine" if name == "cosine" else name,
                mean,
            )
            remaining = record["remaining_chunks"][record["valid"].bool()].float()
            row = (distance, remaining)
            if str(record["problem_id"]) in train_ids:
                train_rows.append(row)
            else:
                test_rows.append(row)
                test_distances.append(distance)
            by_problem.setdefault(str(record["problem_id"]), []).append((record, distance))
        metric = progress_metrics(test_distances)
        metric["remaining_chunk_r2"] = remaining_r2(train_rows, test_rows)
        # Matched-depth candidate branches use a correct trajectory's terminal
        # as the explicitly oracle goal for both branches.
        correct_values, incorrect_values = [], []
        grouped: dict[str, list[dict]] = {}
        for record in records:
            grouped.setdefault(str(record["problem_id"]), []).append(record)
        for problem_records in grouped.values():
            correct = next((r for r in problem_records if r["correct"]), None)
            incorrect = next((r for r in problem_records if not r["correct"]), None)
            if (correct is None or incorrect is None
                    or str(correct["problem_id"]) in train_ids):
                continue
            goal = correct["states"][int(correct["terminal_index"])].float()
            depth = min(int(correct["valid"].sum()), int(incorrect["valid"].sum())) // 2
            if depth < 1:
                continue
            c_state = correct["states"][depth].float()
            i_state = incorrect["states"][depth].float()
            if name == "cosine":
                c_distance = 1 - torch.nn.functional.cosine_similarity(c_state, goal, dim=0)
                i_distance = 1 - torch.nn.functional.cosine_similarity(i_state, goal, dim=0)
            else:
                if mean is not None:
                    c_state, i_state, goal = c_state - mean, i_state - mean, goal - mean
                if transform is not None:
                    c_state, i_state, goal = c_state @ transform, i_state @ transform, goal @ transform
                c_distance = (c_state - goal).norm()
                i_distance = (i_state - goal).norm()
            correct_values.append(c_distance)
            incorrect_values.append(i_distance)
        metric["matched_depth_candidate_pairs"] = len(correct_values)
        metric["matched_depth_candidate_accuracy"] = (
            candidate_ranking_accuracy(
                torch.stack(correct_values), torch.stack(incorrect_values)
            ) if correct_values else float("nan")
        )
        results[name] = metric
    report = {
        "schema_version": 1,
        "kind": "oracle_terminal_state_geometry_probe",
        "features": str(args.features),
        "metadata": payload["metadata"],
        "split": "problem_hash_80_train_20_test",
        "results": results,
        "oracle_terminal_states": True,
        "candidate_privileged_outcomes": True,
        "usable_at_inference": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
