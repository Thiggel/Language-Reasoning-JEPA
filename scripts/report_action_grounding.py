"""Report the matched action--outcome alignment falsifier.

The shuffled condition permutes observed on-trajectory intent phrases while
leaving the rendered state/outcome trace and optimization recipe unchanged.
It is a causal diagnostic of whether correct action identity is needed, not a
candidate model or a fixed-point component ablation.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ROWS = [
    ("aligned intent--outcome pairs", "disc_latent_goal_h2_r1"),
    (
        "exact-paired permuted intents",
        "disc_latent_goal_h2_r2_shuffled_actions_paired",
    ),
]


def payload(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def planning(run: Path, slack: int) -> float | None:
    data = payload(run / f"plan_slack{slack}_look1.json")
    return next(
        (float(value["success"]) for key, value in data.items()
         if key.startswith("latent_planner")),
        None,
    )


def number(value: float | None) -> str:
    return "---" if value is None else f"{value:.3f}"


def main() -> None:
    lines = [
        "# Observed-action grounding falsifier",
        "",
        "The intervention permutes the on-trajectory intent phrases relative",
        "to their rendered transitions while retaining the same architecture,",
        "loss weights, and state/outcome data. It is a falsifier: a substantial",
        "drop supports dependence on correctly aligned observed actions; a tie",
        "would invalidate a strong action-conditioned-dynamics interpretation.",
        "",
        "| training alignment | status | strict | slack-2 | transition match | value tau | clean-history top-1 | after-error top-1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, name in ROWS:
        run = RUNS / name
        audit = payload(run / "counterfactual_audit.json")
        failures = payload(run / "failures.json").get("report", {})
        decisions = failures.get("one_step_decisions", {})
        status = (
            "complete" if (run / "DONE").exists()
            else "failed" if (run / "FAILED").exists()
            else "training"
        )
        lines.append(
            f"| {label} | {status} | {number(planning(run, 0))} | "
            f"{number(planning(run, 2))} | {number(audit.get('match'))} | "
            f"{number(audit.get('tau_value'))} | "
            f"{number(decisions.get('clean_history_top1'))} | "
            f"{number(decisions.get('off_history_top1'))} |"
        )

    destination = RUNS / "action_grounding.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
