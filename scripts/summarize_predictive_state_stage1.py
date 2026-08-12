#!/usr/bin/env python3
"""Produce a compact, validity-aware Stage 1 evaluation summary."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    evaluations = metrics.get("evaluations", {})
    if not evaluations:
        raise ValueError("training metrics contain no held-out evaluation")
    final_step = max(evaluations, key=lambda value: int(value))
    final = evaluations[final_step]
    scalars = {
        key: value for key, value in final.items()
        if isinstance(value, (int, float))
    }
    finite = all(math.isfinite(float(value)) for value in scalars.values())
    action_gap = None
    if "permuted_action_cosine_loss" in final:
        action_gap = (
            final["permuted_action_cosine_loss"]
            - final["transition_cosine_loss"]
        )
    geometry = final.get("target_geometry", {})
    frozen = metrics["backbone_mode"] == "frozen"
    if not finite:
        validity, claim = "invalid_nonfinite", "Non-finite held-out metrics."
    elif frozen:
        validity = "diagnostic_only"
        claim = (
            "Operational frozen-backbone diagnostic only; no representation "
            "or recurrent-decoding claim."
        )
    else:
        # A single adapted cell cannot establish Stage 1 on its own: the gate
        # is a cross-cell contrast against the token- and capacity-matched
        # NTP-only, no-action, and action-only arms.
        validity = "not_admitted_pending_controls"
        claim = (
            "Single upper-half LoRA cell; no representation claim until the "
            "matched NTP-only, no-action, and action-only arms are compared."
        )
    summary = {
        "schema_version": 1,
        "status": "completed",
        "process_status": "COMPLETED",
        "run_id": os.environ.get("RUN_ID"),
        "variant": metrics["variant"],
        "backbone_mode": metrics["backbone_mode"],
        "source_revision": metrics.get("source_revision"),
        "model_id": metrics["model_id"],
        "model_revision": metrics["model_revision"],
        "evaluation_step": int(final_step),
        "held_out": final,
        "action_permutation_loss_gap": action_gap,
        "finite_metrics": finite,
        "effective_rank": geometry.get("effective_rank"),
        "scientific_validity": validity,
        "claim": claim,
        "oracle_information": False,
        "candidate_privileged_information": False,
        "cross_project_information": False,
        "artifacts": [
            "model/best.pt", "model/last.pt", "model/metrics.json",
            "eval_summary.json", "run_summary.json",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
