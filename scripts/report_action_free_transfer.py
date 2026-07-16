"""Compare inferred-action sentence-stream models across text domains."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CELLS = (
    ("stylized iGSM", "online stop-gradient", "SIGReg", "off", "svjepa_noldad_sg_sigreg"),
    ("stylized iGSM", "online stop-gradient", "SIGReg", "on", "svjepa_ldad_sg_sigreg"),
    ("official iGSM", "EMA", "VICReg", "off", "sentence_vjepa_faithful"),
    ("official iGSM", "online stop-gradient", "SIGReg", "off", "sentence_vjepa_faithful_sg_sigreg"),
    ("official iGSM", "online stop-gradient", "SIGReg", "on", "sentence_vjepa_faithful_sg_sigreg_ldad"),
)
PRIOR_CELLS = (
    ("svjepa_noldad_sg_sigreg", 0.1, 4.0),
    ("svjepa_noldad_sg_sigreg_kl0", 0.1, 0.0),
)
PRIOR_ARCHITECTURE_CELLS = (
    ("one Gaussian", 1, "svjepa_noldad_sg_sigreg"),
    ("Gaussian mixture", 4, "svjepa_noldad_sg_sigreg_mix4"),
)
COUNTERFACTUAL_SET_CELLS = (
    ("off", "one Gaussian", 1, "sentence_vjepa_faithful_sg_sigreg"),
    ("on", "one Gaussian", 1, "sentence_vjepa_faithful_cfset_gaussian"),
    ("off", "Gaussian mixture", 4, "sentence_vjepa_faithful_mixture"),
    ("on", "Gaussian mixture", 4, "sentence_vjepa_faithful_cfset_mixture"),
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


def counterfactual(name: str) -> dict:
    source = RUNS / name / "variational_counterfactual_audit.json"
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


def number(data: dict, key: str) -> str:
    try:
        return f"{float(data[key]):.3f}"
    except (KeyError, TypeError, ValueError):
        return "---"


def main() -> None:
    lines = [
        "# Action-free probabilistic transfer",
        "",
        "The posterior observes the next sentence; the prior is available",
        "before it. Prior prediction and shuffled-prior sensitivity are the",
        "primary pre-transition diagnostics. Same-context retrieval alone is",
        "confounded because both codes can identify the unique state.",
        "",
        "| domain | target | regularizer | posterior-code reconstruction | status | state std | state rank | posterior L1 | prior-mean L1 | prior shuffle ratio | prior best-of-8 L1 | prior/post. cosine | retrieval top-1 |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, target, regularizer, ldad, name in CELLS:
        row, data = validation(name), probe(name)
        status = "complete" if data.get("prior_mean_prediction_l1") is not None else (
            "probing" if (RUNS / name / "variational_probe.json").exists()
            else "training"
        )
        lines.append(
            f"| {domain} | {target} | {regularizer} | {ldad} | {status} | "
            f"{metric(row, 'state_std')} | {metric(row, 'state_effrank')} | "
            f"{number(data, 'matched_prediction_l1')} | "
            f"{number(data, 'prior_mean_prediction_l1')} | "
            f"{number(data, 'prior_mean_action_sensitivity_ratio')} | "
            f"{number(data, 'prior_best_of_8_prediction_l1')} | "
            f"{number(data, 'prior_posterior_cosine')} | "
            f"{number(data, 'prior_posterior_retrieval_top1')} |"
        )
    lines.extend([
        "",
        "## Same-state counterfactual coverage",
        "",
        "At each held-out state, every feasible rendered outcome is encoded.",
        "Distinct coverage requires a prior prediction to enter the ball with",
        "radius half the distance to the nearest competing outcome; nearest-",
        "candidate assignment alone is not counted as coverage. Posterior-",
        "accuracy coverage is stricter: a prior sample must predict a candidate",
        "at least as accurately as its outcome-informed posterior mean.",
        "",
        "| domain | target | regularizer | posterior-code reconstruction | true outcome separation | posterior distinct match | prior distinct outcome coverage | prior posterior-accuracy coverage | prior distinct posterior-mode coverage | necessary candidate share | prior necessary assignment |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for domain, target, regularizer, ldad, name in CELLS:
        data = counterfactual(name)
        lines.append(
            f"| {domain} | {target} | {regularizer} | {ldad} | "
            f"{number(data, 'mean_true_candidate_separation_l1')} | "
            f"{number(data, 'posterior_distinct_candidate_match')} | "
            f"{number(data, 'prior_distinct_candidate_coverage')} | "
            f"{number(data, 'prior_within_posterior_error_coverage')} | "
            f"{number(data, 'prior_distinct_posterior_mode_coverage')} | "
            f"{number(data, 'necessary_candidate_fraction')} | "
            f"{number(data, 'prior_necessary_assignment_rate')} |"
        )
    lines.extend([
        "",
        "## Prior-alignment diagnostic",
        "",
        "Both rows use stylized iGSM, online stop-gradient targets, SIGReg,",
        "and no posterior-code reconstruction. Removing the free-nats",
        "allowance tests whether posterior--prior mismatch, rather than",
        "posterior mode quality, limits precise pre-transition coverage.",
        "",
        "| action-KL weight | free nats | status | state std | state rank | posterior L1 | prior-mean L1 | prior shuffle ratio | prior best-of-8 L1 | posterior distinct match | prior distinct coverage | prior posterior-accuracy coverage |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, kl_weight, free_nats in PRIOR_CELLS:
        row, data, cf = validation(name), probe(name), counterfactual(name)
        status = "complete" if data.get("prior_mean_prediction_l1") is not None else (
            "probing" if (RUNS / name / "variational_probe.json").exists()
            else "training"
        )
        lines.append(
            f"| {kl_weight:g} | {free_nats:g} | {status} | "
            f"{metric(row, 'state_std')} | {metric(row, 'state_effrank')} | "
            f"{number(data, 'matched_prediction_l1')} | "
            f"{number(data, 'prior_mean_prediction_l1')} | "
            f"{number(data, 'prior_mean_action_sensitivity_ratio')} | "
            f"{number(data, 'prior_best_of_8_prediction_l1')} | "
            f"{number(cf, 'posterior_distinct_candidate_match')} | "
            f"{number(cf, 'prior_distinct_candidate_coverage')} | "
            f"{number(cf, 'prior_within_posterior_error_coverage')} |"
        )
    lines.extend([
        "",
        "## Prior-architecture diagnostic",
        "",
        "Both rows restore four free nats and leave posterior-code",
        "reconstruction disabled. The only causal change is replacing the",
        "single diagonal-Gaussian plan-time prior with four Gaussian mixture",
        "components. Component usage is reported to distinguish useful",
        "multimodality from component collapse.",
        "",
        "| prior | components | status | effective components | max component probability | posterior distinct match | prior distinct coverage | prior posterior-accuracy coverage | prior posterior-mode coverage | best-of-8 L1 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label, components, name in PRIOR_ARCHITECTURE_CELLS:
        data, cf = probe(name), counterfactual(name)
        status = "complete" if data.get("prior_mean_prediction_l1") is not None else (
            "probing" if (RUNS / name / "variational_probe.json").exists()
            else "training"
        )
        lines.append(
            f"| {label} | {components} | {status} | "
            f"{number(data, 'prior_effective_components')} | "
            f"{number(data, 'prior_max_component_probability')} | "
            f"{number(cf, 'posterior_distinct_candidate_match')} | "
            f"{number(cf, 'prior_distinct_candidate_coverage')} | "
            f"{number(cf, 'prior_within_posterior_error_coverage')} | "
            f"{number(cf, 'prior_distinct_posterior_mode_coverage')} | "
            f"{number(data, 'prior_best_of_8_prediction_l1')} |"
        )
    lines.extend([
        "",
        "## Counterfactual outcome-set factorial (official iGSM)",
        "",
        "All rows use online stop-gradient targets, SIGReg, four action-KL",
        "free nats, and no posterior-code reconstruction. Outcome-set rows",
        "train on every feasible rendered next sentence but never receive",
        "intent phrases, action identifiers, relevance, remaining-step counts,",
        "or preference labels. Crossing set supervision with prior capacity",
        "tests whether a mixture becomes useful only when all modes are shown.",
        "",
        "| outcome set | prior | components | status | state std | state rank | posterior distinct match | prior distinct coverage | prior posterior-accuracy coverage | prior posterior-mode coverage | effective components | max component probability | best-of-8 L1 |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for outcome_set, prior, components, name in COUNTERFACTUAL_SET_CELLS:
        row, data, cf = validation(name), probe(name), counterfactual(name)
        status = "complete" if data.get("prior_mean_prediction_l1") is not None else (
            "probing" if (RUNS / name / "variational_probe.json").exists()
            else "training" if (RUNS / name).exists()
            else "pending"
        )
        lines.append(
            f"| {outcome_set} | {prior} | {components} | {status} | "
            f"{metric(row, 'state_std')} | {metric(row, 'state_effrank')} | "
            f"{number(cf, 'posterior_distinct_candidate_match')} | "
            f"{number(cf, 'prior_distinct_candidate_coverage')} | "
            f"{number(cf, 'prior_within_posterior_error_coverage')} | "
            f"{number(cf, 'prior_distinct_posterior_mode_coverage')} | "
            f"{number(data, 'prior_effective_components')} | "
            f"{number(data, 'prior_max_component_probability')} | "
            f"{number(data, 'prior_best_of_8_prediction_l1')} |"
        )
    destination = RUNS / "action_free_transfer.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
