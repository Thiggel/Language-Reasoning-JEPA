"""Report the exact stylized-to-official transfer of probabilistic raw LDAD."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CELLS = (
    ("stylized iGSM", "SIGReg", "pooled", "residual", "dvldad_ema_sig_off", "off"),
    ("stylized iGSM", "SIGReg", "pooled", "residual", "dvldad_ema_sig_on", "on"),
    ("official iGSM", "SIGReg", "pooled", "residual", "dvjepa_faithful_pooled_ema_sigreg", "off"),
    ("official iGSM", "SIGReg", "pooled", "residual", "dvjepa_faithful_pooled_ema_sigreg_ldad", "on"),
    ("official iGSM", "VICReg", "pooled", "residual", "dvjepa_faithful_pooled_ema_vicreg", "off"),
    ("official iGSM", "VICReg", "pooled", "residual", "dvjepa_faithful_pooled_ema_vicreg_ldad", "on"),
    ("official iGSM", "SIGReg", "pooled", "direct", "dvjepa_faithful_pooled_ema_sigreg_direct", "off"),
    ("official iGSM", "SIGReg", "pooled", "direct", "dvjepa_faithful_pooled_ema_sigreg_direct_ldad", "on"),
    ("official iGSM", "VICReg", "pooled", "direct", "dvjepa_faithful_pooled_ema_vicreg_direct", "off"),
    ("official iGSM", "VICReg", "pooled", "direct", "dvjepa_faithful_pooled_ema_vicreg_direct_ldad", "on"),
    ("official iGSM", "SIGReg", "ordered tokens", "residual", "dvjepa_faithful_token2_ema_sigreg", "off"),
    ("official iGSM", "SIGReg", "ordered tokens", "residual", "dvjepa_faithful_token2_ema_sigreg_ldad", "on"),
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
        "# Probabilistic raw-action LDAD transfer",
        "",
        "All cells use EMA targets, 30k fresh examples per epoch, 10 epochs,",
        "and a 16-dimensional observed-action code. SIGReg rows use weight",
        "0.01; VICReg rows use weight 1. Residual pooled rows test cross-domain",
        "raw-action LDAD transfer and regularization; direct pairs isolate",
        "transition parameterization; ordered-token rows",
        "isolate whether preserving word order improves conditioning.",
        "",
        "| domain | regularizer | action representation | predictor | raw LDAD | status | state std | state rank | action rank | matched L1 | shuffled/matched | residual^2 | token acc. | exact phrase |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, regularizer, action, predictor, name, ldad in CELLS:
        row, payload = validation(name), probe(name)
        status = "complete" if payload else (
            "training" if (RUNS / name / "last.pt").exists() else "pending"
        )
        lines.append(
            f"| {domain} | {regularizer} | {action} | {predictor} | {ldad} | {status} | "
            f"{metric(row, 'state_std')} | {metric(row, 'state_effrank')} | "
            f"{number(payload, 'observed_action_effective_rank')} | "
            f"{number(payload, 'matched_prediction_l1')} | "
            f"{number(payload, 'action_sensitivity_ratio')} | "
            f"{number(payload, 'calibration', 'mean_standardized_residual_squared')} | "
            f"{metric(row, 'observed_action_token_accuracy')} | "
            f"{metric(row, 'observed_action_sequence_exact')} |"
        )
    destination = RUNS / "variational_transfer.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
