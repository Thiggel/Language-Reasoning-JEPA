"""Create a live, component-aware report for recipe screens and ablations.

Example::

    .venv/bin/python scripts/recipe_report.py \
      --glob 'disc_gar_*' \
      --reference disc_gar_greedy_h2_k2_ldad \
      --out runs/recipe_screen.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from omegaconf import OmegaConf


def nested(cfg, path: str, default=None):
    cur = cfg
    for key in path.split("."):
        if cur is None or key not in cur:
            return default
        cur = cur[key]
    return cur


def weight(cfg, name: str) -> float:
    return float(nested(cfg, f"objective.{name}.weight", 0.0))


def plan_success(run: Path, slack: int) -> float | None:
    path = run / f"plan_slack{slack}_look1.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for key, metrics in data.items():
        if key.startswith("latent_planner"):
            return float(metrics["success"])
    return None


def validation_epochs(run: Path) -> int:
    path = run / "metrics.csv"
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(
            bool(row.get("val/loss")) for row in csv.DictReader(handle)
        )


def audit_metric(run: Path, name: str) -> float | None:
    path = run / "counterfactual_audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(name)


def fmt(value, digits=3):
    return "---" if value is None else f"{value:.{digits}f}"


def matched_reference_name(reference: str, run_name: str) -> str:
    """Return the seed-matched reference for a conventional ``_sN`` run."""
    match = re.search(r"(_s\d+)$", run_name)
    return f"{reference}{match.group(1) if match else ''}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--glob", default="disc_gar_*")
    parser.add_argument(
        "--reference", default="disc_gar_greedy_h2_k2_ldad"
    )
    parser.add_argument("--out", default="runs/recipe_screen.md")
    args = parser.parse_args()
    root = Path(args.runs)
    rows = []
    for run in sorted(root.glob(args.glob)):
        config_path = run / ".hydra" / "config.yaml"
        if not config_path.exists():
            continue
        cfg = OmegaConf.load(config_path)
        strict, slack = plan_success(run, 0), plan_success(run, 2)
        epochs = validation_epochs(run)
        total_epochs = int(nested(cfg, "train.epochs", 0))
        log_path = Path(f"runs_{run.name}.log")
        error = ""
        if log_path.exists():
            match = re.findall(
                r"(OutOfMemoryError|Error executing job|Traceback|Killed)",
                log_path.read_text(errors="replace"),
            )
            error = match[-1] if match else ""
        status = (
            "complete" if strict is not None and slack is not None
            else "failed" if error
            else "evaluating" if epochs >= total_epochs
            else f"train {epochs}/{total_epochs}"
        )
        rows.append({
            "run": run.name,
            "status": status,
            "policy": nested(cfg, "data.geo_rank_policy", "---"),
            "H": int(nested(cfg, "data.geo_rank_horizon", 1)),
            "K": int(nested(cfg, "data.geo_rank_k", 0)),
            "B": int(nested(cfg, "data.geo_rank_beam_width", 1)),
            "alt": int(nested(cfg, "data.n_alt", 0)),
            "latent": weight(cfg, "latent_pred"),
            "LDAD": weight(cfg, "observed_action_ldad"),
            "mono": weight(cfg, "monotone"),
            "mono_margin": float(
                nested(cfg, "objective.monotone.margin", 0.0)
            ),
            "gar_gap": float(
                nested(cfg, "objective.geo_rank.label_gap", 0.0)
            ),
            "distill": weight(cfg, "value_distill"),
            "CF": weight(cfg, "counterfactual_outcome"),
            "hier": weight(cfg, "hierarchy"),
            "roll": weight(cfg, "rollout"),
            "anchor": weight(cfg, "chunk_pred"),
            "anchor_roll": float(
                nested(cfg, "objective.chunk_pred.rollout_weight", 0.0)
            ),
            "geo": weight(cfg, "geo_rank"),
            "VIC": weight(cfg, "vicreg"),
            "res": bool(nested(cfg, "model.predictor_residual", True)),
            "target": nested(cfg, "model.state_target", "ema"),
            "action": nested(cfg, "model.action_encoder_kind", "pooled"),
            "strict": strict,
            "slack2": slack,
            "match": audit_metric(run, "match"),
            "tau": audit_metric(run, "tau_value"),
        })

    # Round-specific globs often contain only ablations/additions and omit
    # the reference directory itself.  Load references independently and
    # preserve matched-seed comparisons for conventional ``_sN`` run names.
    reference_success = {
        ("", slack): plan_success(root / args.reference, slack)
        for slack in (0, 2)
    }

    def matched_reference(run_name: str, slack: int) -> float | None:
        reference_name = matched_reference_name(args.reference, run_name)
        suffix = reference_name[len(args.reference):]
        key = (suffix, slack)
        if key not in reference_success:
            reference_success[key] = plan_success(
                root / reference_name, slack
            )
        return reference_success[key]
    lines = [
        "# Minimal-recipe screen",
        "",
        "Strict-budget success is primary; slack-2 is secondary. A dash means",
        "the post-training evaluation has not completed.",
        "",
        "| run | status | preference teacher | H | K | B | label gap | latent | LDAD | monotonicity (w/m) | energy regression | counterfactual transition | hierarchy | rollout | outcome | outcome-roll | geometric preference | VICReg | target | action encoder | residual | @strict | Δstrict | @slack2 | Δslack2 | match | τ(value) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        reference_strict = matched_reference(row["run"], 0)
        reference_slack = matched_reference(row["run"], 2)
        strict_delta = (
            None if reference_strict is None or row["strict"] is None
            else row["strict"] - reference_strict
        )
        slack_delta = (
            None if reference_slack is None or row["slack2"] is None
            else row["slack2"] - reference_slack
        )
        lines.append(
            f"| {row['run']} | {row['status']} | {row['policy']} | {row['H']} | {row['K']} | {row['B']} | "
            f"{row['gar_gap']:g} | {row['latent']:g} | {row['LDAD']:g} | "
            f"{row['mono']:g}/{row['mono_margin']:g} | {row['distill']:g} | {row['CF']:g} | "
            f"{row['hier']:g} | {row['roll']:g} | {row['anchor']:g} | "
            f"{row['anchor_roll']:g} | {row['geo']:g} | {row['VIC']:g} | "
            f"{row['target']} | {row['action']} | "
            f"{'yes' if row['res'] else 'no'} | {fmt(row['strict'])} | "
            f"{fmt(strict_delta)} | {fmt(row['slack2'])} | "
            f"{fmt(slack_delta)} | {fmt(row['match'])} | {fmt(row['tau'])} |"
        )
    destination = Path(args.out)
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
