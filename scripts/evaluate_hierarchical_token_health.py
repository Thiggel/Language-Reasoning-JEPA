#!/usr/bin/env python3
"""Held-out token dynamics and causal shortcut health gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.analysis.hierarchical_language import predictor_shortcut_diagnostics
from textjepa.data.provenance import sha256_file
from textjepa.utils.language_planning_runtime import load_hierarchical_checkpoint


@torch.no_grad()
def main():
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
    forward = model.token_forward(
        features["hidden_states"][:count].to(args.device, dtype=dtype),
        features["input_ids"][:count].to(args.device),
        attention_mask=features["attention_mask"][:count].to(args.device),
        prompt_len=features["prompt_len"][:count].to(args.device),
        solution_end=features["solution_end"][:count].to(args.device),
    )
    diagnostics = predictor_shortcut_diagnostics(
        model.p0, forward["token_states"], forward["token_actions"],
        forward["token_targets"], forward["token_valid"],
        learner.token_metric,
    )
    full = learner.token_metric(
        forward["token_predictions"], forward["token_targets"]
    )[forward["token_valid"]]
    output = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset_fingerprint": features["dataset_fingerprint"],
        "metrics": {"heldout_token_dynamics": float(full.mean()), **diagnostics},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
