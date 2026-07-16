"""Summarize the controlled probabilistic discourse/action experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CELLS = (
    ("dvjepa_latent_sg_sigreg", "inferred latent", "---"),
    ("dvjepa_pooled_sg_sigreg", "pooled observed intent", "off"),
    ("dvjepa_pooled_sg_sigreg_ldad", "pooled observed intent", "on"),
    ("dvjepa_token_sg_sigreg", "token-concatenated intent", "off"),
    ("dvjepa_token_sg_sigreg_ldad", "token-concatenated intent", "on"),
)


def last_validation(path: Path) -> dict[str, str]:
    source = path / "metrics.csv"
    if not source.exists():
        return {}
    rows = [row for row in csv.DictReader(source.open()) if row.get("val/loss")]
    return rows[-1] if rows else {}


def metric(row: dict[str, str], key: str) -> str:
    try:
        return f"{float(row[f'val/{key}']):.3f}"
    except (KeyError, TypeError, ValueError):
        return "---"


def probe(path: Path) -> dict:
    source = path / "variational_probe.json"
    if not source.exists():
        return {}
    try:
        return json.loads(source.read_text())
    except json.JSONDecodeError:
        return {}


def number(payload: dict, *keys: str) -> str:
    current = payload
    try:
        for key in keys:
            current = current[key]
        return f"{float(current):.3f}"
    except (KeyError, TypeError, ValueError):
        return "---"


def first_available(*values: str) -> str:
    return next((value for value in values if value != "---"), "---")


def main() -> None:
    lines = [
        "# Probabilistic discourse JEPA: controlled action observability",
        "",
        "All cells use online stop-gradient, SIGReg 0.01, 30k fresh examples",
        "per epoch, and 10 epochs. The next state is diagonal Gaussian in all",
        "cells. Only action observability/encoder and raw-action LDAD differ.",
        "",
        "| action conditioning | raw LDAD | status | state std | state rank | action std | action rank | pred std | target std | standardized residual² | 1σ / 2σ coverage | action sensitivity | action op probe | action value probe | prior value probe | prior→posterior top-1 | displacement value probe | token acc. | exact phrase |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label, ldad in CELLS:
        path = RUNS / name
        row, payload = last_validation(path), probe(path)
        status = "complete" if payload else (
            "training" if (path / "last.pt").exists() else "pending"
        )
        source = "posterior" if label == "inferred latent" else "observed_action"
        lines.append(
            f"| {label} | {ldad} | {status} | "
            f"{metric(row, 'state_std')} | {metric(row, 'state_effrank')} | "
            f"{first_available(number(payload, f'{source}_std'), metric(row, 'action_q_mu_std'))} | "
            f"{first_available(number(payload, f'{source}_effective_rank'), metric(row, 'action_q_mu_effrank'))} | "
            f"{metric(row, 'pred_std')} | {metric(row, 'target_std')} | "
            f"{number(payload, 'calibration', 'mean_standardized_residual_squared')} | "
            f"{number(payload, 'calibration', 'coverage_1sigma')} / "
            f"{number(payload, 'calibration', 'coverage_2sigma')} | "
            f"{number(payload, 'action_sensitivity_ratio')} | "
            f"{number(payload, 'linear_probe_accuracy', source, 'op')} | "
            f"{number(payload, 'linear_probe_accuracy', source, 'value')} | "
            f"{number(payload, 'linear_probe_accuracy', 'prior', 'value')} | "
            f"{number(payload, 'prior_posterior_retrieval_top1')} | "
            f"{number(payload, 'linear_probe_accuracy', 'delta', 'value')} | "
            f"{metric(row, 'observed_action_token_accuracy')} | "
            f"{metric(row, 'observed_action_sequence_exact')} |"
        )
    destination = RUNS / "discourse_variational.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
