"""Report the observed-action representation and LDAD bottleneck sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CELLS = (
    ("dvjepa_pooled_sg_sigreg", "mean-pooled sentence", "---", "off"),
    ("dvjepa_pooled_sg_sigreg_ldad", "mean-pooled sentence", "---", "on"),
    ("dvjepa_token2_sg_sigreg", "ordered token concatenation", "2", "off"),
    ("dvjepa_token2_sg_sigreg_ldad", "ordered token concatenation", "2", "on"),
    ("dvjepa_token4_sg_sigreg", "ordered token concatenation", "4", "off"),
    ("dvjepa_token4_sg_sigreg_ldad", "ordered token concatenation", "4", "on"),
    ("dvjepa_token_sg_sigreg", "ordered token concatenation", "8", "off"),
    ("dvjepa_token_sg_sigreg_ldad", "ordered token concatenation", "8", "on"),
)


def validation(name: str) -> dict[str, str]:
    source = RUNS / name / "metrics.csv"
    if not source.exists():
        return {}
    rows = [r for r in csv.DictReader(source.open()) if r.get("val/loss")]
    return rows[-1] if rows else {}


def probe(name: str) -> dict:
    source = RUNS / name / "variational_probe.json"
    return json.loads(source.read_text()) if source.exists() else {}


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
        "# Observed-action token bottleneck sweep",
        "",
        "All cells use the same probabilistic next-state JEPA, online",
        "stop-gradient, SIGReg 0.01, 30k fresh examples per epoch, and a",
        "16-dimensional final action code. Token width is the dimension kept",
        "per ordered token before concatenation and final projection.",
        "",
        "| action representation | token width | raw LDAD | status | state rank | action rank | matched L1 | shuffled/matched | action op | displacement op | displacement value | standardized residual² | token accuracy | exact phrase |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, representation, width, ldad in CELLS:
        row, result = validation(name), probe(name)
        status = "complete" if result else ("training" if row else "pending")
        lines.append(
            f"| {representation} | {width} | {ldad} | {status} | "
            f"{metric(row, 'state_effrank')} | "
            f"{number(result, 'observed_action_effective_rank')} | "
            f"{number(result, 'matched_prediction_l1')} | "
            f"{number(result, 'action_sensitivity_ratio')} | "
            f"{number(result, 'linear_probe_accuracy', 'observed_action', 'op')} | "
            f"{number(result, 'linear_probe_accuracy', 'delta', 'op')} | "
            f"{number(result, 'linear_probe_accuracy', 'delta', 'value')} | "
            f"{number(result, 'calibration', 'mean_standardized_residual_squared')} | "
            f"{metric(row, 'observed_action_token_accuracy')} | "
            f"{metric(row, 'observed_action_sequence_exact')} |"
        )
    destination = RUNS / "action_encoder_sweep.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
