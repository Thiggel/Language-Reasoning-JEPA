"""Report horizon and root-candidate sweeps for latent-goal preferences."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CELLS = (
    ("disc_gar_greedy_h1_k2_ldad", 1, 2),
    ("disc_gar_greedy_h2_k2_ldad", 2, 2),
    ("disc_gar_greedy_h4_k2_ldad", 4, 2),
    ("disc_gar_greedy_h8_k2_ldad", 8, 2),
    ("disc_gar_greedy_h16_k2_ldad", 16, 2),
    ("disc_gar_greedy_h2_k4_ldad", 2, 4),
    ("disc_gar_greedy_h2_k8_ldad", 2, 8),
)


def payload(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def number(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "---"


def plan(run: Path, slack: int) -> str:
    result = payload(run / f"plan_slack{slack}_look1.json")
    return number(result.get("latent_planner", {}).get("success"))


def main() -> None:
    lines = [
        "# Multi-step latent-goal preference sweep",
        "",
        "H is the environment continuation horizon used to construct a",
        "geometric preference. K is the number of alternative root actions",
        "sampled during training; inference still scores every feasible action.",
        "Teacher diagnostics compare learned geometric labels to exact",
        "remaining-computation order only for analysis.",
        "",
        "| H | K | status | teacher top-1 | pair coverage | pair tau-a | emitted tie rate | strict | slack-2 | student value tau |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, horizon, alternatives in CELLS:
        run = RUNS / name
        teacher = payload(run / "gar_teacher_audit.json").get(
            "teacher_vs_oracle", {}
        )
        counterfactual = payload(run / "counterfactual_audit.json")
        complete = (run / "plan_slack0_look1.json").exists()
        status = "complete" if complete else (
            "training" if (run / "last.pt").exists() else "pending"
        )
        lines.append(
            f"| {horizon} | {alternatives} | {status} | "
            f"{number(teacher.get('top1_accuracy'))} | "
            f"{number(teacher.get('oracle_pair_coverage'))} | "
            f"{number(teacher.get('oracle_pair_tau_a'))} | "
            f"{number(teacher.get('emitted_oracle_tie_rate'))} | "
            f"{plan(run, 0)} | {plan(run, 2)} | "
            f"{number(counterfactual.get('tau_value'))} |"
        )
    lines.extend([
        "",
        "## Fixed-checkpoint teacher horizon audit",
        "",
        "This diagnostic holds the reduced H=2/K=2 checkpoint fixed and",
        "recomputes only the geometry-greedy continuation labels. It",
        "separates intrinsic teacher quality from student retraining.",
        "",
        "| H | status | teacher top-1 | pair coverage | pair tau-a | emitted tie rate |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    reference = RUNS / "disc_latent_goal_h2_r1"
    for horizon in (1, 2, 4, 8, 16):
        data = payload(reference / f"gar_teacher_h{horizon}.json")
        teacher = data.get("teacher_vs_oracle", {})
        status = "complete" if teacher else "pending"
        lines.append(
            f"| {horizon} | {status} | "
            f"{number(teacher.get('top1_accuracy'))} | "
            f"{number(teacher.get('oracle_pair_coverage'))} | "
            f"{number(teacher.get('oracle_pair_tau_a'))} | "
            f"{number(teacher.get('emitted_oracle_tie_rate'))} |"
        )
    lines.extend([
        "",
        "## Fixed-checkpoint bounded-beam continuation audit",
        "",
        "This paired 200-anchor diagnostic retains B lowest-distance",
        "continuations per root. It uses the same target geometry and no",
        "symbolic quality signal; B=1 is the original greedy teacher.",
        "",
        "| H | beam B | status | teacher top-1 | pair coverage | pair tau-a | decisive accuracy | emitted tie rate |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for horizon, beam in (
        (2, 1), (4, 1), (4, 2), (4, 4), (4, 8),
        (8, 1), (8, 4), (8, 8), (16, 8),
    ):
        data = payload(reference / f"gar_teacher_h{horizon}_b{beam}.json")
        teacher = data.get("teacher_vs_oracle", {})
        status = "complete" if teacher else "pending"
        lines.append(
            f"| {horizon} | {beam} | {status} | "
            f"{number(teacher.get('top1_accuracy'))} | "
            f"{number(teacher.get('oracle_pair_coverage'))} | "
            f"{number(teacher.get('oracle_pair_tau_a'))} | "
            f"{number(teacher.get('oracle_pair_accuracy_decisive'))} | "
            f"{number(teacher.get('emitted_oracle_tie_rate'))} |"
        )
    destination = RUNS / "preference_sweep.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
