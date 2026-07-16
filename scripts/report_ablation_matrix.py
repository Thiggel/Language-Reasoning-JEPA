"""Report seed-matched component removals around a selected recipe.

The report never mixes seeds: a run ending in ``_s2`` is compared with the
reference ``_s2`` checkpoint, and similarly for seed 3.  Live rows state the
number of complete matched pairs so two-seed selection evidence cannot be
mistaken for a final three-seed paper result.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


DEFAULT_COMPARISONS = {
    "latent-state prediction": "disc_latent_goal_h2_r2_nolatent",
    "on-trajectory outcome anchoring": "disc_latent_goal_h2_r2_nooutcome",
    "predicted-outcome consistency": "disc_latent_goal_h2_r2_nooutroll",
    "latent-goal preference distillation": "disc_latent_goal_h2_r2_nopref",
    "variance--covariance regularization": "disc_latent_goal_h2_r2_novic",
    "EMA target network": "disc_latent_goal_h2_r2_noema",
}
SEED_SUFFIXES = ("", "_s2", "_s3")


def success(root: Path, run: str, slack: int) -> float | None:
    source = root / run / f"plan_slack{slack}_look1.json"
    if not source.exists():
        return None
    payload = json.loads(source.read_text())
    return next(
        (
            float(metrics["success"])
            for key, metrics in payload.items()
            if key.startswith("latent_planner")
        ),
        None,
    )


def matched_pairs(
    root: Path, reference: str, comparison: str, slack: int
) -> tuple[list[float], list[float]]:
    reference_values, comparison_values = [], []
    for suffix in SEED_SUFFIXES:
        ref = success(root, reference + suffix, slack)
        cmp = success(root, comparison + suffix, slack)
        if ref is None or cmp is None:
            continue
        reference_values.append(ref)
        comparison_values.append(cmp)
    return reference_values, comparison_values


def parse_comparison(spec: str) -> tuple[str, str]:
    try:
        label, run = spec.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "comparison must be LABEL=RUN"
        ) from exc
    if not label or not run:
        raise argparse.ArgumentTypeError("comparison must be LABEL=RUN")
    return label, run


def fmt(values: list[float]) -> str:
    if not values:
        return "---"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def fmt_loss(reference: list[float], comparison: list[float]) -> str:
    if not reference or len(reference) != len(comparison):
        return "---"
    return f"{statistics.mean(reference) - statistics.mean(comparison):+.3f}"


def exact(values: list[float]) -> str:
    return " / ".join(f"{value:.3f}" for value in values) or "---"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--reference", default="disc_latent_goal_h2_r1")
    parser.add_argument("--out", default="runs/component_removal_matrix.md")
    parser.add_argument(
        "--comparison", action="append", type=parse_comparison,
        help="repeat LABEL=RUN for a custom component matrix",
    )
    args = parser.parse_args()
    root = Path(args.runs)
    comparisons = (
        dict(args.comparison) if args.comparison else DEFAULT_COMPARISONS
    )

    lines = [
        "# Seed-matched component-removal matrix",
        "",
        f"Reference: `{args.reference}`. Positive deltas are accuracy losses",
        "from removing the named component. Every mean uses exactly the same",
        "available seed suffixes on both sides; `3/3` is required for the",
        "final paper table.",
        "",
        "| component removed | run | matched seeds | strict | loss | slack-2 | loss |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    details = []
    for label, run in comparisons.items():
        ref_strict, cmp_strict = matched_pairs(root, args.reference, run, 0)
        ref_slack, cmp_slack = matched_pairs(root, args.reference, run, 2)
        if len(cmp_strict) != len(cmp_slack):
            raise ValueError(
                f"{run}: strict/slack matched-seed counts differ "
                f"({len(cmp_strict)} vs {len(cmp_slack)})"
            )
        n = len(cmp_strict)
        lines.append(
            f"| {label} | `{run}` | {n}/3 | {fmt(cmp_strict)} | "
            f"{fmt_loss(ref_strict, cmp_strict)} | {fmt(cmp_slack)} | "
            f"{fmt_loss(ref_slack, cmp_slack)} |"
        )
        details.append((label, run, ref_strict, cmp_strict, ref_slack, cmp_slack))

    lines.extend([
        "",
        "## Exact matched values",
        "",
        "| component removed | reference strict | removal strict | reference slack-2 | removal slack-2 |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, _run, ref_strict, cmp_strict, ref_slack, cmp_slack in details:
        lines.append(
            f"| {label} | {exact(ref_strict)} | {exact(cmp_strict)} | "
            f"{exact(ref_slack)} | {exact(cmp_slack)} |"
        )

    destination = Path(args.out)
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
