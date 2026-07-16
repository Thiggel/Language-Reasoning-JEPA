"""Report the observed-action probabilistic JEPA LDAD factorial."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TARGETS = (("ema", "EMA target"), ("sg", "online stop-gradient"),
           ("grad", "online gradients"))
REGULARIZERS = (("none", "none"), ("vic", "VICReg"), ("sig", "SIGReg"))


def validation(path: Path) -> dict[str, str]:
    source = path / "metrics.csv"
    if not source.exists():
        return {}
    rows = [row for row in csv.DictReader(source.open()) if row.get("val/loss")]
    return rows[-1] if rows else {}


def probe(path: Path) -> dict:
    source = path / "variational_probe.json"
    if not source.exists():
        return {}
    try:
        return json.loads(source.read_text())
    except json.JSONDecodeError:
        return {}


def metric(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[f"val/{key}"])
    except (KeyError, TypeError, ValueError):
        return None


def nested(payload: dict, *keys: str) -> float | None:
    current = payload
    try:
        for key in keys:
            current = current[key]
        return float(current)
    except (KeyError, TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    return "---" if value is None else f"{value:.3f}"


def cell(target: str, regularizer: str, ldad: str) -> tuple[dict, dict, str]:
    path = RUNS / f"dvldad_{target}_{regularizer}_{ldad}"
    row, payload = validation(path), probe(path)
    status = "complete" if (path / "DONE").exists() and payload else (
        "training" if (path / "last.pt").exists() else "pending"
    )
    return row, payload, status


def main() -> None:
    lines = [
        "# Faithful observed-action variational LDAD factorial",
        "",
        "All cells use the same probabilistic next-state discourse JEPA, a",
        "pooled observed intent action, 30k fresh examples per epoch, and 10",
        "epochs. Only target mode, anti-collapse regularizer, and raw-token",
        "LDAD differ. This is distinct from posterior-code reconstruction",
        "in the action-free model, which has no external action target.",
        "",
        "| target | regularizer | raw LDAD | status | state std | state rank | matched L1 | shuffled/matched | standardized residual^2 | 1-sigma / 2-sigma | displacement op | displacement value | token acc. | exact phrase |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, target_label in TARGETS:
        for regularizer, regularizer_label in REGULARIZERS:
            for ldad in ("off", "on"):
                row, payload, status = cell(target, regularizer, ldad)
                lines.append(
                    f"| {target_label} | {regularizer_label} | {ldad} | {status} | "
                    f"{fmt(metric(row, 'state_std'))} | "
                    f"{fmt(metric(row, 'state_effrank'))} | "
                    f"{fmt(nested(payload, 'matched_prediction_l1'))} | "
                    f"{fmt(nested(payload, 'action_sensitivity_ratio'))} | "
                    f"{fmt(nested(payload, 'calibration', 'mean_standardized_residual_squared'))} | "
                    f"{fmt(nested(payload, 'calibration', 'coverage_1sigma'))} / "
                    f"{fmt(nested(payload, 'calibration', 'coverage_2sigma'))} | "
                    f"{fmt(nested(payload, 'linear_probe_accuracy', 'delta', 'op'))} | "
                    f"{fmt(nested(payload, 'linear_probe_accuracy', 'delta', 'value'))} | "
                    f"{fmt(metric(row, 'observed_action_token_accuracy'))} | "
                    f"{fmt(metric(row, 'observed_action_sequence_exact'))} |"
                )

    lines += [
        "",
        "## Paired effect of raw-action LDAD (on minus off)",
        "",
        "Negative matched-L1 deltas and positive sensitivity deltas indicate",
        "better action-conditioned probabilistic prediction. Calibration and",
        "state geometry remain separate diagnostics.",
        "",
        "| target | regularizer | delta state std | delta state rank | delta matched L1 | delta shuffled/matched | delta displacement value |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for target, target_label in TARGETS:
        for regularizer, regularizer_label in REGULARIZERS:
            off_row, off_payload, _ = cell(target, regularizer, "off")
            on_row, on_payload, _ = cell(target, regularizer, "on")
            pairs = (
                (metric(on_row, "state_std"), metric(off_row, "state_std")),
                (metric(on_row, "state_effrank"), metric(off_row, "state_effrank")),
                (nested(on_payload, "matched_prediction_l1"), nested(off_payload, "matched_prediction_l1")),
                (nested(on_payload, "action_sensitivity_ratio"), nested(off_payload, "action_sensitivity_ratio")),
                (nested(on_payload, "linear_probe_accuracy", "delta", "value"), nested(off_payload, "linear_probe_accuracy", "delta", "value")),
            )
            deltas = [None if a is None or b is None else a - b for a, b in pairs]
            lines.append(
                f"| {target_label} | {regularizer_label} | "
                + " | ".join(fmt(value) for value in deltas) + " |"
            )

    destination = RUNS / "observed_vjepa_ldad_factorial.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
