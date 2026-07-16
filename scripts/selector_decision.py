"""Apply the predeclared bounded-continuation selector gate.

The current H=2/B=1 teacher is the complexity reference.  A longer bounded
beam is admissible only when neither planning budget degrades by more than the
tie threshold and at least one improves beyond it.  A one-seed gain through
the replication threshold requests a matched seed; a larger gain advances to
the next fixed-point removal round.  After two matched seeds, an admissible
mean gain above the tie threshold advances without recursive replication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CANDIDATES = {
    "H=4, beam=8": "disc_latent_goal_h4_beam8_r1",
    "H=8, beam=8": "disc_latent_goal_h8_beam8_r1",
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


def parse_candidate(spec: str) -> tuple[str, str]:
    try:
        label, run = spec.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "candidate must be LABEL=RUN"
        ) from exc
    if not label or not run:
        raise argparse.ArgumentTypeError("candidate must be LABEL=RUN")
    return label, run


def decision(
    strict_gain: float,
    slack_gain: float,
    n_seeds: int,
    tie_threshold: float = 0.02,
    replication_threshold: float = 0.05,
) -> str:
    if min(strict_gain, slack_gain) < -tie_threshold - 1e-12:
        return "reject"
    best = max(strict_gain, slack_gain)
    if best <= tie_threshold + 1e-12:
        return "retain H=2/B=1"
    if n_seeds >= 2 or best > replication_threshold + 1e-12:
        return "advance"
    return "second seed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--reference", default="disc_latent_goal_h2_r1")
    parser.add_argument("--out", default="runs/selector_decision.md")
    parser.add_argument("--candidate", action="append", type=parse_candidate)
    parser.add_argument("--tie-threshold", type=float, default=0.02)
    parser.add_argument("--replication-threshold", type=float, default=0.05)
    args = parser.parse_args()
    if args.tie_threshold > args.replication_threshold:
        parser.error("tie threshold cannot exceed replication threshold")

    root = Path(args.runs)
    candidates = dict(args.candidate) if args.candidate else CANDIDATES
    ref_strict = success(root, args.reference, 0)
    ref_slack = success(root, args.reference, 2)
    lines = [
        "# Bounded-continuation selector decision",
        "",
        f"Complexity reference: `{args.reference}` (H=2, beam=1).",
        "Strict-budget accuracy is primary and slack-2 is a conservative",
        "gate. Reject if either budget loses more than",
        f"{args.tie_threshold:g}; retain the simpler teacher if neither gain",
        f"exceeds {args.tie_threshold:g}. Replicate a one-seed gain through",
        f"{args.replication_threshold:g}; larger gains advance immediately.",
        "After two matched seeds, any admissible mean gain above the tie",
        "threshold advances to a new fixed-point removal round.",
        "",
        "| continuation teacher | run | strict | gain | slack-2 | gain | decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for label, run in candidates.items():
        n_reference, n_run = seed_count(args.reference), seed_count(run)
        if n_reference != n_run:
            raise ValueError(
                "matched comparison requires equal seed counts: "
                f"reference has {n_reference}, {label!r} has {n_run}"
            )
        strict, slack = success(root, run, 0), success(root, run, 2)
        if None in (ref_strict, ref_slack, strict, slack):
            fields = ("---", "---", "---", "---", "pending")
        else:
            strict_gain = strict - ref_strict
            slack_gain = slack - ref_slack
            fields = (
                f"{strict:.3f}", f"{strict_gain:+.3f}",
                f"{slack:.3f}", f"{slack_gain:+.3f}",
                decision(
                    strict_gain, slack_gain, n_run,
                    args.tie_threshold, args.replication_threshold,
                ),
            )
        lines.append(
            f"| {label} | `{run}` | {fields[0]} | {fields[1]} | "
            f"{fields[2]} | {fields[3]} | {fields[4]} |"
        )

    destination = Path(args.out)
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
