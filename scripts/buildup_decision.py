"""Apply the predeclared component-addition rule to a recipe build-up.

An addition is useful only when it improves at least one planning budget by
more than ``gain_threshold`` and degrades neither budget by more than that
threshold. A one-seed best gain in the ambiguity band requests one matched
replication; a larger gain advances immediately to the combined candidate.
After two matched seeds, any admissible mean gain above the threshold advances
without recursively requesting more seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ADDITIONS = {
    "faithful raw-action reconstruction": "real_latent_goal_h2_addldad",
    "residual transition parameterization": "real_latent_goal_h2_addresidual",
    "open-loop latent prediction": "real_latent_goal_h2_addrollout",
    "macro-transition prediction": "real_latent_goal_h2_addhierarchy",
    "counterfactual transition augmentation": "real_latent_goal_h2_addcfout",
    "geometry-to-energy regression": "real_latent_goal_h2_adddistill",
}


def success(root: Path, run_spec: str, slack: int) -> float | None:
    values = []
    for name in run_spec.split(","):
        source = root / name.strip() / f"plan_slack{slack}_look1.json"
        if not source.exists():
            return None
        payload = json.loads(source.read_text())
        value = next(
            (float(metrics["success"]) for key, metrics in payload.items()
             if key.startswith("latent_planner")),
            None,
        )
        if value is None:
            return None
        values.append(value)
    return sum(values) / len(values) if values else None


def seed_count(run_spec: str) -> int:
    return len([name for name in run_spec.split(",") if name.strip()])


def parse_addition(spec: str) -> tuple[str, str]:
    try:
        component, run = spec.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "addition must be COMPONENT=RUN"
        ) from exc
    if not component or not run:
        raise argparse.ArgumentTypeError("addition must be COMPONENT=RUN")
    return component, run


def decision(
    strict_gain: float,
    slack_gain: float,
    n_seeds: int,
    gain_threshold: float = 0.02,
    replication_threshold: float = 0.05,
) -> str:
    if min(strict_gain, slack_gain) < -gain_threshold - 1e-12:
        return "reject"
    best = max(strict_gain, slack_gain)
    if best <= gain_threshold + 1e-12:
        return "reject"
    if n_seeds >= 2 or best > replication_threshold + 1e-12:
        return "add"
    return "second seed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--reference", default="real_latent_goal_h2_r1")
    parser.add_argument("--out", default="runs/official_build_decision.md")
    parser.add_argument("--addition", action="append", type=parse_addition)
    parser.add_argument("--gain-threshold", type=float, default=0.02)
    parser.add_argument("--replication-threshold", type=float, default=0.05)
    args = parser.parse_args()
    if args.gain_threshold > args.replication_threshold:
        parser.error("gain threshold cannot exceed replication threshold")

    root = Path(args.runs)
    additions = dict(args.addition) if args.addition else ADDITIONS
    reference_strict = success(root, args.reference, 0)
    reference_slack = success(root, args.reference, 2)
    lines = [
        "# Official-iGSM component build-up decision",
        "",
        f"Reference: `{args.reference}`. A positive number is an accuracy",
        "gain from adding the component. Reject if either budget loses more",
        f"than {args.gain_threshold:g} or neither gain exceeds that threshold.",
        f"Replicate a one-seed best gain through {args.replication_threshold:g};",
        "after two matched seeds, an admissible mean gain above the threshold",
        "advances to the combined candidate.",
        "",
        "| component added | run | strict | gain | slack-2 | gain | decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for component, run in additions.items():
        n_reference = seed_count(args.reference)
        n_run = seed_count(run)
        if n_reference != n_run:
            raise ValueError(
                "matched comparison requires equal seed counts: "
                f"reference has {n_reference}, {component!r} has {n_run}"
            )
        strict = success(root, run, 0)
        slack = success(root, run, 2)
        if None in (reference_strict, reference_slack, strict, slack):
            fields = ("---", "---", "---", "---", "pending")
        else:
            strict_gain = strict - reference_strict
            slack_gain = slack - reference_slack
            fields = (
                f"{strict:.3f}", f"{strict_gain:+.3f}",
                f"{slack:.3f}", f"{slack_gain:+.3f}",
                decision(
                    strict_gain, slack_gain, n_run,
                    args.gain_threshold, args.replication_threshold,
                ),
            )
        lines.append(
            f"| {component} | `{run}` | {fields[0]} | {fields[1]} | "
            f"{fields[2]} | {fields[3]} | {fields[4]} |"
        )

    destination = Path(args.out)
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
