"""Report the pure two-objective text-domain Delta-JEPA controls."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CELLS = (
    ("deltajepa_text_noldad", "latent prediction only", "off", "---"),
    ("deltajepa_text_h1", "adjacent displacement", "on", "1"),
    ("deltajepa_text_h4", "long-horizon displacement", "on", "4"),
)


def validation(name: str) -> dict[str, str]:
    source = RUNS / name / "metrics.csv"
    if not source.exists():
        return {}
    rows = [row for row in csv.DictReader(source.open()) if row.get("val/loss")]
    return rows[-1] if rows else {}


def audit(name: str) -> dict:
    source = RUNS / name / "counterfactual_audit.json"
    return json.loads(source.read_text()) if source.exists() else {}


def metric(row: dict[str, str], key: str) -> str:
    try:
        return f"{float(row[f'val/{key}']):.3f}"
    except (KeyError, TypeError, ValueError):
        return "---"


def number(payload: dict, key: str) -> str:
    try:
        return f"{float(payload[key]):.3f}"
    except (KeyError, TypeError, ValueError):
        return "---"


def main() -> None:
    lines = [
        "# Pure text-domain Delta-JEPA controls",
        "",
        "All cells use fully online gradients and unnormalized latent MSE,",
        "with no EMA, stop-gradient, VICReg, SIGReg, outcome anchor, planning",
        "energy, hierarchy, or open-loop auxiliary. H-step LDAD reconstructs",
        "H ordered externally observed intent phrases from s_{t+H}-s_t.",
        "",
        "| model | raw-action LDAD | H | status | state std | state rank | token accuracy | exact phrase | transition match | RSA |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label, ldad, horizon in CELLS:
        row, result = validation(name), audit(name)
        status = "complete" if result else ("training" if row else "pending")
        lines.append(
            f"| {label} | {ldad} | {horizon} | {status} | "
            f"{metric(row, 'state_std')} | {metric(row, 'state_effrank')} | "
            f"{metric(row, 'observed_action_token_accuracy')} | "
            f"{metric(row, 'observed_action_sequence_exact')} | "
            f"{number(result, 'match')} | {number(result, 'rsa')} |"
        )
    destination = RUNS / "delta_jepa_text.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
