"""Summarize the 18-cell inferred-action sentence-stream VJEPA matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TARGETS = (("ema", "EMA"), ("sg", "online stop-gradient"),
           ("online", "online gradients"))
REGS = (("none", "none"), ("vicreg", "VICReg"), ("sigreg", "SIGReg"))
LDADS = (("noldad", "off"), ("ldad", "on"))


def _last_validation(path: Path) -> dict[str, str]:
    source = path / "metrics.csv"
    if not source.exists():
        return {}
    rows = list(csv.DictReader(source.open()))
    rows = [row for row in rows if row.get("val/loss")]
    return rows[-1] if rows else {}


def _fmt(row: dict[str, str], key: str) -> str:
    try:
        return f"{float(row[f'val/{key}']):.3f}"
    except (KeyError, TypeError, ValueError):
        return "---"


def _probe(name: str) -> dict:
    source = RUNS / name / "variational_probe.json"
    if not source.exists():
        return {}
    return json.loads(source.read_text())


def _number(payload: dict, *keys: str) -> str:
    value = payload
    try:
        for key in keys:
            value = value[key]
        return f"{float(value):.3f}"
    except (KeyError, TypeError, ValueError):
        return "---"


def main() -> None:
    lines = [
        "# Inferred-action sentence-stream VJEPA factorial",
        "",
        "This is the action-free variational matrix. Its displacement decoder",
        "targets the learned posterior action mean rather than an externally",
        "observed action. We therefore call it posterior-code reconstruction,",
        "not LDAD or a faithful Delta-JEPA replication.",
        "",
        "| target | regularizer | posterior-code reconstruction | state std | state rank | posterior mean std | posterior rank | prior mean std | predicted std | target std |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, target_label in TARGETS:
        for reg, reg_label in REGS:
            for ldad, ldad_label in LDADS:
                name = f"svjepa_{ldad}_{target}_{reg}"
                row = _last_validation(RUNS / name)
                lines.append(
                    f"| {target_label} | {reg_label} | {ldad_label} | "
                    f"{_fmt(row, 'state_std')} | {_fmt(row, 'state_effrank')} | "
                    f"{_fmt(row, 'action_q_mu_std')} | "
                    f"{_fmt(row, 'action_q_mu_effrank')} | "
                    f"{_fmt(row, 'action_p_mu_std')} | "
                    f"{_fmt(row, 'pred_std')} | {_fmt(row, 'target_std')} |"
                )
    lines.extend([
        "",
        "## Semantic diagnostic for the strongest no-EMA pair",
        "",
        "Both rows use online stop-gradient and SIGReg. The posterior sees the",
        "next state, whereas the prior does not; therefore posterior semantics",
        "alone do not establish a usable pre-transition action representation.",
        "",
        "| posterior-code reconstruction | matched L1 | shuffled/matched error | posterior rank | posterior op | posterior value | prior op | prior value | displacement op | displacement value | prior→posterior top-1 | MRR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for ldad, label in (("noldad", "off"), ("ldad", "on")):
        probe = _probe(f"svjepa_{ldad}_sg_sigreg")
        lines.append(
            f"| {label} | {_number(probe, 'matched_prediction_l1')} | "
            f"{_number(probe, 'action_sensitivity_ratio')} | "
            f"{_number(probe, 'posterior_effective_rank')} | "
            f"{_number(probe, 'linear_probe_accuracy', 'posterior', 'op')} | "
            f"{_number(probe, 'linear_probe_accuracy', 'posterior', 'value')} | "
            f"{_number(probe, 'linear_probe_accuracy', 'prior', 'op')} | "
            f"{_number(probe, 'linear_probe_accuracy', 'prior', 'value')} | "
            f"{_number(probe, 'linear_probe_accuracy', 'delta', 'op')} | "
            f"{_number(probe, 'linear_probe_accuracy', 'delta', 'value')} | "
            f"{_number(probe, 'prior_posterior_retrieval_top1')} | "
            f"{_number(probe, 'prior_posterior_retrieval_mrr')} |"
        )
    destination = RUNS / "variational_factorial.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
