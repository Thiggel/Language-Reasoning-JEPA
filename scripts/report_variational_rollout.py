"""Compare open-loop uncertainty growth for observed-action V-JEPA pairs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CELLS = (
    ("stylized", "off", "dvldad_ema_sig_off"),
    ("stylized", "on", "dvldad_ema_sig_on"),
    ("official", "off", "dvjepa_faithful_pooled_ema_sigreg"),
    ("official", "on", "dvjepa_faithful_pooled_ema_sigreg_ldad"),
)


def payload(name: str) -> dict:
    source = RUNS / name / "variational_rollout.json"
    if not source.exists():
        return {}
    try:
        return json.loads(source.read_text())
    except json.JSONDecodeError:
        return {}


def transfer_payload(name: str) -> dict:
    source = RUNS / name / "variational_rollout_test_calibrated.json"
    if not source.exists():
        return {}
    try:
        return json.loads(source.read_text())
    except json.JSONDecodeError:
        return {}


def number(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "---"


def main() -> None:
    sample_counts = sorted({
        data.get("rollout_samples")
        for _, _, name in CELLS
        if (data := payload(name)).get("rollout_samples") is not None
    })
    sample_text = (
        f"{sample_counts[0]} Gaussian samples are propagated recursively."
        if len(sample_counts) == 1
        else "Gaussian samples are propagated recursively."
    )
    lines = [
        "# Observed-action open-loop uncertainty growth",
        "",
        "The true intent sequence is fixed while " + sample_text,
        "Teacher-forced error re-encodes the true",
        "history at every step; open-loop error and spread expose accumulated",
        "bias and uncertainty. Values use target-encoder means.",
        "",
        "| domain | raw-action reconstruction | horizon | n | teacher-forced L1 | open-loop L1 | open-loop std | standardized residual² | coverage 1σ | coverage 2σ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, ldad, name in CELLS:
        data = payload(name)
        horizons = data.get("by_horizon", {})
        if not horizons:
            lines.append(
                f"| {domain} | {ldad} | --- | --- | --- | --- | --- | --- | --- | --- |"
            )
            continue
        for horizon in (1, 2, 4, 8):
            row = horizons.get(str(horizon))
            if not row:
                continue
            lines.append(
                f"| {domain} | {ldad} | {horizon} | {row['n']} | "
                f"{number(row.get('teacher_forced_l1'))} | "
                f"{number(row.get('open_loop_l1'))} | "
                f"{number(row.get('open_loop_std'))} | "
                f"{number(row.get('open_loop_z2'))} | "
                f"{number(row.get('open_loop_coverage_1sigma'))} | "
                f"{number(row.get('open_loop_coverage_2sigma'))} |"
            )
    lines.extend([
        "",
        "## Held-out scalar spread calibration",
        "",
        "For each horizon, one variance temperature is fitted on the first",
        "half of validation trajectories and evaluated on the disjoint second",
        "half. A calibrated residual squared near 1 and two-sigma coverage",
        "near 0.954 indicate that a scalar spread correction is sufficient;",
        "persistent error instead indicates non-scalar rollout misspecification.",
        "",
        "| domain | raw-action reconstruction | horizon | temperature | raw residual² | calibrated residual² | raw coverage 1σ/2σ | calibrated coverage 1σ/2σ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for domain, ldad, name in CELLS:
        data = payload(name)
        for horizon in (1, 2, 4, 8):
            row = data.get("by_horizon", {}).get(str(horizon), {})
            if row.get("variance_temperature") is None:
                continue
            lines.append(
                f"| {domain} | {ldad} | {horizon} | "
                f"{number(row.get('variance_temperature'))} | "
                f"{number(row.get('evaluation_z2_raw'))} | "
                f"{number(row.get('evaluation_z2_calibrated'))} | "
                f"{number(row.get('evaluation_coverage_1sigma_raw'))}/"
                f"{number(row.get('evaluation_coverage_2sigma_raw'))} | "
                f"{number(row.get('evaluation_coverage_1sigma_calibrated'))}/"
                f"{number(row.get('evaluation_coverage_2sigma_calibrated'))} |"
            )
    lines.extend([
        "",
        "## Frozen validation-to-test calibration transfer",
        "",
        "The temperature fitted above on validation is applied unchanged to",
        "every trajectory in a disjoint generated test corpus. No test residual",
        "is used to select or refit the temperature.",
        "",
        "| domain | raw-action reconstruction | horizon | validation temperature | test residual² raw/calibrated | test coverage 1σ raw/calibrated | test coverage 2σ raw/calibrated |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for domain, ldad, name in CELLS:
        data = transfer_payload(name)
        for horizon in (1, 2, 4, 8):
            row = data.get("by_horizon", {}).get(str(horizon), {})
            if row.get("external_variance_temperature") is None:
                continue
            lines.append(
                f"| {domain} | {ldad} | {horizon} | "
                f"{number(row.get('external_variance_temperature'))} | "
                f"{number(row.get('transfer_z2_raw'))}/"
                f"{number(row.get('transfer_z2_calibrated'))} | "
                f"{number(row.get('transfer_coverage_1sigma_raw'))}/"
                f"{number(row.get('transfer_coverage_1sigma_calibrated'))} | "
                f"{number(row.get('transfer_coverage_2sigma_raw'))}/"
                f"{number(row.get('transfer_coverage_2sigma_calibrated'))} |"
            )
    destination = RUNS / "variational_rollout.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
