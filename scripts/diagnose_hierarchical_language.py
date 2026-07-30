#!/usr/bin/env python3
"""Run representation, probe, shortcut, and counterfactual diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.analysis.hierarchical_language import (
    counterfactual_generation_diagnostics,
    linear_probe_diagnostics,
    representation_statistics,
    shortcut_error_ratios,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _json(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {name: _json(item) for name, item in value.items()}
    return value


def main() -> None:
    args = parse_args()
    payload = torch.load(
        args.artifact, map_location="cpu", weights_only=True
    )
    if "representations" not in payload:
        raise ValueError("diagnostic artifact requires representations")
    result = {"representations": {}}
    for name, states in payload["representations"].items():
        result["representations"][name] = representation_statistics(states)
    if "probes" in payload:
        result["probes"] = {}
        for name, probe in payload["probes"].items():
            result["probes"][name] = linear_probe_diagnostics(
                probe["train_states"], probe["train_targets"],
                probe["test_states"], probe["test_targets"],
                classification=bool(probe["classification"]),
                ridge=float(probe.get("ridge", 1e-3)),
            )
    if "shortcut_errors" in payload:
        errors = payload["shortcut_errors"]
        result["shortcuts"] = shortcut_error_ratios(
            errors["full"],
            **{
                name: value for name, value in errors.items()
                if name != "full"
            },
        )
    if "counterfactuals" in payload:
        result["counterfactuals"] = counterfactual_generation_diagnostics(
            **payload["counterfactuals"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
