#!/usr/bin/env python3
"""Collapse and shortcut health for token-only or joint nested JEPA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.analysis.hierarchical_language import (
    predictor_shortcut_diagnostics,
    representation_statistics,
)
from textjepa.data.provenance import sha256_file
from textjepa.utils.language_planning_runtime import load_hierarchical_checkpoint


def _summary(prefix: str, states: torch.Tensor) -> dict[str, float]:
    statistics = representation_statistics(states)
    std = statistics["std"]
    eigenvalues = statistics["eigenvalues"]
    return {
        f"{prefix}_mean_coordinate_std": float(std.mean()),
        f"{prefix}_minimum_coordinate_std": float(std.min()),
        f"{prefix}_participation_ratio": float(
            statistics["participation_ratio"]
        ),
        f"{prefix}_entropic_effective_rank": float(
            statistics["entropic_effective_rank"]
        ),
        f"{prefix}_leading_eigenvalue_fraction": float(
            eigenvalues[0] / eigenvalues.sum().clamp_min(1e-12)
        ),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    model, learner = load_hierarchical_checkpoint(args.checkpoint, args.device)
    count = min(args.max_examples, len(features["hidden_states"]))
    dtype = next(model.parameters()).dtype
    hidden = features["hidden_states"][:count].to(args.device, dtype=dtype)
    token_ids = features["input_ids"][:count].to(args.device)
    attention = features["attention_mask"][:count].to(args.device)
    prompt_len = features["prompt_len"][:count].to(args.device)
    solution_end = features["solution_end"][:count].to(args.device)
    token = model.token_forward(
        hidden,
        token_ids,
        attention_mask=attention,
        prompt_len=prompt_len,
        solution_end=solution_end,
    )
    token_shortcuts = predictor_shortcut_diagnostics(
        model.p0,
        token["token_states"],
        token["token_actions"],
        token["token_targets"],
        token["token_valid"],
        learner.token_metric,
    )
    metrics = {
        **{f"token_{name}": value for name, value in token_shortcuts.items()},
        **_summary("token", token["token_states"][token["token_valid"]]),
    }
    experiment = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    ).get("experiment_config") or {}
    joint = bool(
        experiment.get("research", {}).get("joint_token_sentence", False)
    )
    if joint:
        sentence = model.sentence_forward(
            hidden,
            token_ids,
            features["boundaries"][:count].to(args.device),
            attention_mask=attention,
            prompt_len=prompt_len,
            solution_end=solution_end,
        )
        sentence_shortcuts = predictor_shortcut_diagnostics(
            model.p1,
            sentence["sentence_states"],
            sentence["sentence_actions"],
            sentence["sentence_targets"],
            sentence["sentence_valid"],
            learner.sentence_metric,
        )
        metrics.update({
            **{
                f"sentence_{name}": value
                for name, value in sentence_shortcuts.items()
            },
            **_summary(
                "sentence",
                sentence["sentence_states"][sentence["sentence_valid"]],
            ),
        })
    payload = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset_fingerprint": features["dataset_fingerprint"],
        "joint_token_sentence": joint,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
