"""Report controlled probabilistic architecture transfers with EMA targets."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CELLS = (
    ("dvldad_ema_sig_off", "SIGReg", "mean-pooled intent", "residual", "off"),
    ("dvldad_ema_sig_on", "SIGReg", "mean-pooled intent", "residual", "on"),
    ("dvjepa_token2_ema_sigreg", "SIGReg", "ordered-token bottleneck", "residual", "off"),
    ("dvjepa_token2_ema_sigreg_ldad", "SIGReg", "ordered-token bottleneck", "residual", "on"),
    ("dvjepa_pooled_ema_sigreg_direct", "SIGReg", "mean-pooled intent", "direct", "off"),
    ("dvjepa_pooled_ema_sigreg_direct_ldad", "SIGReg", "mean-pooled intent", "direct", "on"),
    ("dvldad_ema_vic_off", "VICReg", "mean-pooled intent", "residual", "off"),
    ("dvldad_ema_vic_on", "VICReg", "mean-pooled intent", "residual", "on"),
    ("dvjepa_pooled_ema_vicreg_direct", "VICReg", "mean-pooled intent", "direct", "off"),
    ("dvjepa_pooled_ema_vicreg_direct_ldad", "VICReg", "mean-pooled intent", "direct", "on"),
)
ONLINE_CELLS = (
    ("dvldad_grad_sig_off", "residual", "off"),
    ("dvldad_grad_sig_on", "residual", "on"),
    ("dvjepa_pooled_grad_sigreg_direct", "direct", "off"),
    ("dvjepa_pooled_grad_sigreg_direct_ldad", "direct", "on"),
)


def validation(name: str) -> dict[str, str]:
    source = RUNS / name / "metrics.csv"
    if not source.exists():
        return {}
    rows = [row for row in csv.DictReader(source.open()) if row.get("val/loss")]
    return rows[-1] if rows else {}


def probe(name: str) -> dict:
    source = RUNS / name / "variational_probe.json"
    if not source.exists():
        return {}
    try:
        return json.loads(source.read_text())
    except json.JSONDecodeError:
        return {}


def metric(row: dict[str, str], key: str) -> str:
    try:
        return f"{float(row[f'val/{key}']):.3f}"
    except (KeyError, TypeError, ValueError):
        return "---"


def number(payload: dict, *keys: str) -> str:
    value = payload
    try:
        for key in keys:
            value = value[key]
        return f"{float(value):.3f}"
    except (KeyError, TypeError, ValueError):
        return "---"


def main() -> None:
    lines = [
        "# Probabilistic architecture transfer with EMA targets",
        "",
        "Every cell uses a 16-dimensional observed-action code, 30k fresh",
        "examples per epoch, and 10 epochs. Ordered tokens retain two",
        "dimensions per word before concatenation. The direct predictor",
        "removes the residual state skip.",
        "",
        "| regularizer | action representation | predictor | raw LDAD | status | state std | state rank | action rank | matched L1 | shuffled/matched | residual^2 | token acc. | exact phrase |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, regularizer, action, predictor, ldad in CELLS:
        row, payload = validation(name), probe(name)
        status = "complete" if payload else (
            "training" if (RUNS / name / "last.pt").exists() else "pending"
        )
        lines.append(
            f"| {regularizer} | {action} | {predictor} | {ldad} | {status} | "
            f"{metric(row, 'state_std')} | {metric(row, 'state_effrank')} | "
            f"{number(payload, 'observed_action_effective_rank')} | "
            f"{number(payload, 'matched_prediction_l1')} | "
            f"{number(payload, 'action_sensitivity_ratio')} | "
            f"{number(payload, 'calibration', 'mean_standardized_residual_squared')} | "
            f"{metric(row, 'observed_action_token_accuracy')} | "
            f"{metric(row, 'observed_action_sequence_exact')} |"
        )
    lines.extend([
        "",
        "## Fully online-gradient SIGReg control",
        "",
        "This isolates whether direct next-state prediction can replace the",
        "EMA target in the strongest no-EMA regularizer setting.",
        "",
        "| predictor | raw LDAD | status | state std | state rank | action rank | matched L1 | shuffled/matched | residual^2 | token acc. | exact phrase |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, predictor, ldad in ONLINE_CELLS:
        row, payload = validation(name), probe(name)
        status = "complete" if payload else (
            "training" if (RUNS / name / "last.pt").exists() else "pending"
        )
        lines.append(
            f"| {predictor} | {ldad} | {status} | "
            f"{metric(row, 'state_std')} | {metric(row, 'state_effrank')} | "
            f"{number(payload, 'observed_action_effective_rank')} | "
            f"{number(payload, 'matched_prediction_l1')} | "
            f"{number(payload, 'action_sensitivity_ratio')} | "
            f"{number(payload, 'calibration', 'mean_standardized_residual_squared')} | "
            f"{metric(row, 'observed_action_token_accuracy')} | "
            f"{metric(row, 'observed_action_sequence_exact')} |"
        )
    destination = RUNS / "variational_architecture.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
