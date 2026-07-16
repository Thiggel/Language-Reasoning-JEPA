"""Report controlled attempts to sharpen non-symbolic action selection."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CELLS = (
    ("reference, seed 1", "disc_latent_goal_h2_r1"),
    ("reference, seed 2", "disc_latent_goal_h2_r1_s2"),
    ("smaller geometry filter", "disc_latent_goal_h2_r1_gap005"),
    ("preference margin 0.25, seed 1", "disc_latent_goal_h2_r1_margin025"),
    ("preference margin 0.25, seed 2", "disc_latent_goal_h2_r1_margin025_s2"),
    ("preference margin 1.0", "disc_latent_goal_h2_r1_margin1"),
    ("preference margin 2.0", "disc_latent_goal_h2_r1_margin2"),
    ("more off-trajectory demonstrations", "disc_latent_goal_h2_r1_distractor30"),
    ("geometry-to-value regression, seed 1", "disc_latent_goal_h2_r2_adddistill"),
    ("geometry-to-value regression, seed 2", "disc_latent_goal_h2_r2_adddistill_s2"),
    ("H=4/B=8 bounded-beam preference teacher", "disc_latent_goal_h4_beam8_r1"),
    ("H=8/B=8 bounded-beam preference teacher", "disc_latent_goal_h8_beam8_r1"),
)


def nested(cfg, path: str, default=None):
    value = cfg
    for key in path.split("."):
        if value is None or key not in value:
            return default
        value = value[key]
    return value


def payload(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def first_payload(*paths: Path) -> dict:
    for path in paths:
        data = payload(path)
        if data:
            return data
    return {}


def success(run: Path, slack: int) -> float | None:
    data = payload(run / f"plan_slack{slack}_look1.json")
    for key, metrics in data.items():
        if key.startswith("latent_planner"):
            return float(metrics["success"])
    return None


def validation_epochs(run: Path) -> tuple[int, int]:
    config = run / ".hydra" / "config.yaml"
    metrics = run / "metrics.csv"
    if not config.exists():
        return 0, 0
    total = int(nested(OmegaConf.load(config), "train.epochs", 0))
    if not metrics.exists():
        return 0, total
    with metrics.open() as handle:
        complete = sum(bool(row.get("val/loss")) for row in csv.DictReader(handle))
    return complete, total


def number(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "---"


def main() -> None:
    lines = [
        "# Non-symbolic selector screen",
        "",
        "Every row uses the reduced direct-predictor model and H=2/K=2",
        "latent-goal continuation preferences. The interventions isolate",
        "label filtering, the learned-energy margin, exposure to distractor",
        "states, and optional geometry-to-value regression.",
        "",
        "| intervention | status | label gap | preference margin | distractor probability | value regression | strict | slack-2 | transition match | value tau | oracle-trace top-1 | clean-history top-1 | after-error top-1 | median useful margin |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, name in CELLS:
        run = RUNS / name
        config_path = run / ".hydra" / "config.yaml"
        cfg = OmegaConf.load(config_path) if config_path.exists() else {}
        strict, slack = success(run, 0), success(run, 2)
        done, total = validation_epochs(run)
        status = "complete" if strict is not None and slack is not None else (
            f"train {done}/{total}" if total else "pending"
        )
        audit = first_payload(
            run / "counterfactual_audit.json",
            run / "fast_counterfactual_audit.json",
        )
        failures = first_payload(
            run / "failures.json", run / "fast_failures.json",
        ).get("report", {}).get(
            "one_step_decisions", {}
        )
        lines.append(
            f"| {label} | {status} | "
            f"{number(nested(cfg, 'objective.geo_rank.label_gap'))} | "
            f"{number(nested(cfg, 'objective.geo_rank.margin'))} | "
            f"{number(nested(cfg, 'data.distractor_prob'))} | "
            f"{number(nested(cfg, 'objective.value_distill.weight'))} | "
            f"{number(strict)} | {number(slack)} | "
            f"{number(audit.get('match'))} | {number(audit.get('tau_value'))} | "
            f"{number(audit.get('top1_value'))} | "
            f"{number(failures.get('clean_history_top1'))} | "
            f"{number(failures.get('off_history_top1'))} | "
            f"{number(failures.get('necessary_margin_median'))} |"
        )
    destination = RUNS / "selector_screen.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
