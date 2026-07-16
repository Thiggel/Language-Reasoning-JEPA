"""Apply the predeclared fixed-point shear rule to completed recipe runs.

The script never launches or deletes an experiment.  It turns the exact
one-component planning results into an auditable decision table:

* remove when both strict and slack-2 losses are at most 0.02;
* on the first seed, obtain a second seed when either loss is in (0.02, 0.05];
* on the first seed, retain when either loss exceeds 0.05;
* after two or more matched seeds, retain whenever the mean loss exceeds
  0.02, so the ambiguity band cannot request replication indefinitely.

Negative losses mean that removing the component improved accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ABLATIONS = {
    "observed-action displacement decoding": "disc_gar_greedy_h2_k2_noldad",
    "goal-distance shaping": "disc_gar_greedy_h2_k2_ldad_nomono",
    "geometry-to-value distillation": "disc_gar_greedy_h2_k2_ldad_nodistill",
    "counterfactual transition prediction": "disc_gar_greedy_h2_k2_ldad_nocfout",
    "macro-transition prediction": "disc_gar_greedy_h2_k2_ldad_nohier",
    "multi-step latent prediction": "disc_gar_greedy_h2_k2_ldad_norollout",
    "on-trajectory outcome prediction": "disc_gar_greedy_h2_k2_ldad_nochunk",
    "variance--covariance regularization": "disc_gar_greedy_h2_k2_ldad_novic",
    "residual transition parameterization": "disc_gar_greedy_h2_k2_ldad_nonres",
    "EMA target network": "disc_gar_greedy_h2_k2_ldad_noema",
}


def success(root: Path, run_spec: str, slack: int) -> float | None:
    """Mean success for a comma-separated set of matched seed runs."""
    values = []
    for run in run_spec.split(","):
        source = root / run.strip() / f"plan_slack{slack}_look1.json"
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


def decision(
    strict_loss: float,
    slack_loss: float,
    n_seeds: int = 1,
    remove_threshold: float = 0.02,
    second_seed_threshold: float = 0.05,
) -> str:
    worst = max(strict_loss, slack_loss)
    if worst <= remove_threshold + 1e-12:
        return "remove"
    if n_seeds >= 2:
        return "retain"
    if worst <= second_seed_threshold + 1e-12:
        return "second seed"
    return "retain"


def seed_count(run_spec: str) -> int:
    return len([run for run in run_spec.split(",") if run.strip()])


def parse_ablation(spec: str) -> tuple[str, str]:
    try:
        component, run = spec.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ablation must be COMPONENT=RUN"
        ) from exc
    if not component or not run:
        raise argparse.ArgumentTypeError("ablation must be COMPONENT=RUN")
    return component, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument(
        "--reference", default="disc_gar_greedy_h2_k2_ldad"
    )
    parser.add_argument("--out", default="runs/shear_decision.md")
    parser.add_argument(
        "--ablation", action="append", type=parse_ablation,
        help=("round-specific COMPONENT=RUN entry; repeat as needed. RUN and "
              "--reference may be comma-separated matched seed runs"),
    )
    parser.add_argument("--remove-threshold", type=float, default=0.02)
    parser.add_argument("--second-seed-threshold", type=float, default=0.05)
    args = parser.parse_args()
    if args.remove_threshold > args.second_seed_threshold:
        parser.error("remove threshold cannot exceed second-seed threshold")
    root = Path(args.runs)
    ablations = dict(args.ablation) if args.ablation else ABLATIONS
    ref_strict = success(root, args.reference, 0)
    ref_slack = success(root, args.reference, 2)
    lines = [
        "# Fixed-point shear decision",
        "",
        f"Reference: `{args.reference}`.",
        "A positive loss means the ablation is worse. Strict-budget accuracy",
        "is primary; slack-2 is a conservative secondary gate. Remove at",
        f"worst loss <= {args.remove_threshold:g}. On one seed, replicate",
        f"through {args.second_seed_threshold:g}; after two matched seeds,",
        "retain any component whose mean loss exceeds the removal threshold.",
        "",
        "| component removed | run | strict | loss | slack-2 | loss | decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for component, run in ablations.items():
        n_ref = seed_count(args.reference)
        n_run = seed_count(run)
        if n_ref != n_run:
            raise ValueError(
                f"matched comparison requires equal seed counts: reference "
                f"has {n_ref}, {component!r} has {n_run}"
            )
        strict, slack = success(root, run, 0), success(root, run, 2)
        if None in (ref_strict, ref_slack, strict, slack):
            fields = ("---", "---", "---", "---", "pending")
        else:
            strict_loss = ref_strict - strict
            slack_loss = ref_slack - slack
            fields = (
                f"{strict:.3f}", f"{strict_loss:+.3f}",
                f"{slack:.3f}", f"{slack_loss:+.3f}",
                decision(
                    strict_loss, slack_loss,
                    n_seeds=n_ref,
                    remove_threshold=args.remove_threshold,
                    second_seed_threshold=args.second_seed_threshold,
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
