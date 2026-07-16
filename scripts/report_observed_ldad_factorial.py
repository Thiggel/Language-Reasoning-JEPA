"""Summarize the faithful deterministic observed-action LDAD factorial."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TARGETS = ("ema", "sg", "grad")
REGS = ("none", "vic", "sig")
LDADS = ("off", "on")
TARGET_LABELS = {
    "ema": "EMA target",
    "sg": "online stop-gradient",
    "grad": "online gradients",
}
REG_LABELS = {"none": "none", "vic": "VICReg", "sig": "SIGReg"}


def _last_validation(path: Path) -> dict[str, str]:
    metrics = path / "metrics.csv"
    if not metrics.exists():
        return {}
    rows = list(csv.DictReader(metrics.open()))
    validation = [row for row in rows if row.get("val/loss")]
    return validation[-1] if validation else {}


def _fmt(row: dict[str, str], key: str) -> str:
    try:
        return f"{float(row[f'val/{key}']):.3f}"
    except (KeyError, TypeError, ValueError):
        return "---"


def _value(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[f"val/{key}"])
    except (KeyError, TypeError, ValueError):
        return None


def _audit(path: Path, key: str) -> str:
    source = path / "counterfactual_audit.json"
    if not source.exists():
        return "---"
    try:
        return f"{float(json.loads(source.read_text())[key]):.3f}"
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "---"


def _audit_value(path: Path, key: str) -> float | None:
    source = path / "counterfactual_audit.json"
    if not source.exists():
        return None
    try:
        return float(json.loads(source.read_text())[key])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _delta(on: float | None, off: float | None) -> str:
    return "---" if on is None or off is None else f"{on - off:+.3f}"


def main() -> None:
    lines = [
        "# Faithful observed-action LDAD stability factorial",
        "",
        "All cells use the same deterministic intent-conditioned JEPA, 30k",
        "fresh training examples per epoch, and 10 epochs. Only target mode,",
        "anti-collapse regularizer, and raw-token LDAD differ.",
        "",
        "| target | regularizer | LDAD | status | state std | state rank | action std | action rank | token acc. | exact phrase | transition match | RSA |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        for reg in REGS:
            for ldad in LDADS:
                name = f"dldad_{target}_{reg}_{ldad}"
                path = RUNS / name
                row = _last_validation(path)
                status = "complete" if (path / "DONE").exists() else (
                    "training" if (path / "last.pt").exists() else "pending"
                )
                lines.append(
                    f"| {TARGET_LABELS[target]} | {REG_LABELS[reg]} | "
                    f"{ldad} | {status} | "
                    f"{_fmt(row, 'state_std')} | {_fmt(row, 'state_effrank')} | "
                    f"{_fmt(row, 'action_std')} | {_fmt(row, 'action_effrank')} | "
                    f"{_fmt(row, 'observed_action_token_accuracy')} | "
                    f"{_fmt(row, 'observed_action_sequence_exact')} | "
                    f"{_audit(path, 'match')} | {_audit(path, 'rsa')} |"
                )
    lines += [
        "",
        "## Paired effect of observed-action LDAD (on minus off)",
        "",
        "Positive matching/RSA deltas mean better transition grounding; state",
        "scale and rank deltas describe geometry and are not accuracy metrics.",
        "",
        "| target | regularizer | Δ state std | Δ state rank | Δ match | Δ RSA |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        for reg in REGS:
            off_path = RUNS / f"dldad_{target}_{reg}_off"
            on_path = RUNS / f"dldad_{target}_{reg}_on"
            off, on = _last_validation(off_path), _last_validation(on_path)
            lines.append(
                f"| {TARGET_LABELS[target]} | {REG_LABELS[reg]} | "
                f"{_delta(_value(on, 'state_std'), _value(off, 'state_std'))} | "
                f"{_delta(_value(on, 'state_effrank'), _value(off, 'state_effrank'))} | "
                f"{_delta(_audit_value(on_path, 'match'), _audit_value(off_path, 'match'))} | "
                f"{_delta(_audit_value(on_path, 'rsa'), _audit_value(off_path, 'rsa'))} |"
            )
    out = RUNS / "observed_ldad_factorial.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
