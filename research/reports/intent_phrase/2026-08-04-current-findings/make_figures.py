#!/usr/bin/env python3
"""Regenerate every quantitative figure in the current intent-JEPA report."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "runs/autonomy/intent_phrase"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

COLORS = {
    "jepa": "#2A6FBB",
    "causal": "#60A5D8",
    "token": "#D98C2B",
    "sentence": "#C94C4C",
    "hybrid": "#8E5BA6",
    "control": "#777777",
    "bad": "#B3261E",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def metric(path: Path) -> tuple[float, float]:
    d = load(path)["metrics_by_slack"]
    return d["0"]["success"], d["2"]["success"]


def summarize(values: list[tuple[float, float]]) -> dict:
    return {
        "strict": [v[0] for v in values],
        "slack2": [v[1] for v in values],
        "strict_mean": statistics.mean(v[0] for v in values),
        "strict_sd": statistics.stdev(v[0] for v in values),
        "slack2_mean": statistics.mean(v[1] for v in values),
        "slack2_sd": statistics.stdev(v[1] for v in values),
        "n_seeds": len(values),
    }


def seed_paths(round_name: str, name_template: str) -> list[Path]:
    return [RUNS / round_name / name_template.format(seed=s) / "metrics.json" for s in range(5)]


def main_results() -> dict:
    paths = {
        "MLP JEPA": seed_paths(
            "2026-07-31-intent-paper300k-jepa-and-ablation-recovery-v2",
            "intent-paper300k-mlp-jepa-s{seed}-v1-recovery-v2",
        ),
        "Causal JEPA": seed_paths(
            "2026-07-31-intent-paper300k-jepa-and-ablation-recovery-v2",
            "intent-paper300k-causal-jepa-s{seed}-v1-recovery-v2",
        ),
        "Token LM": [
            RUNS / "2026-08-03-intent-token-repair-and-ablation-confirm-v1"
            / f"intent-paper300k-token-lm-s{s}-v1-repair-v2" / "metrics.json"
            for s in range(5)
        ],
        "Sentence LM": seed_paths(
            "2026-07-31-intent-paper300k-main-five-seed-v1",
            "intent-paper300k-sentence-lm-s{seed}-v1",
        ),
        "Sentence LM + latent": seed_paths(
            "2026-07-31-intent-paper300k-main-five-seed-v1",
            "intent-paper300k-sentence-latent-lm-s{seed}-v1",
        ),
    }
    return {name: summarize([metric(p) for p in ps]) for name, ps in paths.items()}


def ablation_paths(tag: str) -> list[Path]:
    roots = [
        RUNS / "2026-07-31-intent-paper300k-jepa-and-ablation-recovery-v2",
        RUNS / "2026-08-03-intent-token-repair-and-ablation-confirm-v1",
        RUNS / "2026-08-03-intent-paper300k-ablation-five-seed-completion-v1",
    ]
    by_seed: dict[int, Path] = {}
    for root in roots:
        for p in root.glob(f"*abl-{tag}-s*/metrics.json"):
            match = re.search(r"-s([0-4])(?:-|$)", p.parent.name)
            if match:
                by_seed[int(match.group(1))] = p
    if sorted(by_seed) != list(range(5)):
        raise RuntimeError(f"incomplete ablation {tag}: {sorted(by_seed)}")
    return [by_seed[s] for s in range(5)]


def ablations() -> dict:
    labels = {
        "Full GAR": None,
        "No GAR": "no-gar",
        "Latent MSE only": "mse-only",
        "Ranking only": "rank-only",
        "No counterfactual latent": "no-cf-state",
        "GAR horizon 1": "horizon1",
        "GAR horizon 4": "horizon4",
        "1 counterfactual": "k1",
        "4 counterfactuals": "k4",
        "Advantage MSE 0.10": "mse-w010",
        "Advantage MSE 0.50": "mse-w050",
    }
    main = main_results()["MLP JEPA"]
    out = {}
    for label, tag in labels.items():
        out[label] = main if tag is None else summarize([metric(p) for p in ablation_paths(tag)])
    return out


def mechanism_controls() -> dict:
    round_a = RUNS / "2026-08-03-intent-decision-and-gar-mechanism-audit-v2"
    round_b = RUNS / "2026-08-03-intent-gar-rq3-controls-v2"
    paths = {
        "Full GAR": RUNS / "2026-07-31-intent-paper300k-jepa-and-ablation-recovery-v2"
        / "intent-paper300k-mlp-jepa-s1-v1-recovery-v2/metrics.json",
        "Detached GAR body": round_a / "intent-gar-mech-detached-body-s1-v1/metrics.json",
        "GAR, no CF latent": round_a / "intent-gar-mech-rankmse-no-cfstate-s1-v1/metrics.json",
        "CF latent only": round_a / "intent-gar-mech-cfstate-only-s1-v1/metrics.json",
        "Direct ranker": round_b / "intent-rq3-direct-ranker-lr7e4-s1-v2/metrics.json",
        "Raw geometry only": round_b / "intent-rq3-geometry-only-lr3e4-s1-v2/metrics.json",
    }
    return {name: {"strict": metric(p)[0], "slack2": metric(p)[1], "n_seeds": 1} for name, p in paths.items()}


def representation_results() -> dict:
    main_rounds = {
        "MLP JEPA": (
            "2026-07-31-intent-paper300k-jepa-and-ablation-recovery-v2",
            "intent-paper300k-mlp-jepa-s{seed}-v1-recovery-v2",
        ),
        "Causal JEPA": (
            "2026-07-31-intent-paper300k-jepa-and-ablation-recovery-v2",
            "intent-paper300k-causal-jepa-s{seed}-v1-recovery-v2",
        ),
        "Token LM": (
            "2026-08-03-intent-token-repair-and-ablation-confirm-v1",
            "intent-paper300k-token-lm-s{seed}-v1-repair-v2",
        ),
        "Sentence LM": (
            "2026-07-31-intent-paper300k-main-five-seed-v1",
            "intent-paper300k-sentence-lm-s{seed}-v1",
        ),
        "Sentence LM + latent": (
            "2026-07-31-intent-paper300k-main-five-seed-v1",
            "intent-paper300k-sentence-latent-lm-s{seed}-v1",
        ),
    }
    out = {}
    for name, (round_name, template) in main_rounds.items():
        rows = [load(RUNS / round_name / template.format(seed=s) / "representation_analysis.json") for s in range(5)]
        get = lambda row, *keys: np.asarray([np.nan if x is None else x for x in [
            __import__("functools").reduce(lambda a, k: a[k], keys, row)
        ]], dtype=float)[0]
        metrics = {
            "effective_rank": [get(r, "effective_rank_test") for r in rows],
            "operation_balanced_accuracy": [get(r, "categorical_probes", "operation", "balanced_accuracy") for r in rows],
            "remaining_steps_r2": [get(r, "numeric_probes", "remaining_steps", "r2") for r in rows],
            "resolved_count_r2": [get(r, "numeric_probes", "resolved_count", "r2") for r in rows],
            "outcome_r2": [get(r, "numeric_probes", "outcome_value", "r2") for r in rows],
        }
        out[name] = {k + "_mean": float(np.mean(v)) for k, v in metrics.items()}
        out[name].update({k + "_sd": float(np.std(v, ddof=1)) for k, v in metrics.items()})
    return out


def looped_results() -> dict:
    paths = {
        "Token LM": RUNS / "2026-08-03-intent-loop-eval-recovery-and-gar-policy-v1"
        / "intent-looped300k-token-s0-eval-recovery-v2/metrics.json",
        "Sentence LM": RUNS / "2026-08-03-intent-looped-local-lr-crosscheck-v1"
        / "intent-looped300k-sentence-lr1e3-s0-v1/metrics.json",
        "Sentence LM + latent": RUNS / "2026-08-03-intent-loop-eval-recovery-and-gar-policy-v1"
        / "intent-looped300k-sentence-latent-s0-eval-recovery-v2/metrics.json",
    }
    out = {}
    for name, p in paths.items():
        d = load(p)["metrics_by_loop_and_slack"]
        out[name] = {int(k): {"strict": v["0"]["success"], "slack2": v["2"]["success"]} for k, v in d.items()}
    return out


def scaling_results() -> dict:
    base = RUNS / "2026-08-04-intent-fixed-checkpoint-ttc-pilot-v1"
    labels = {
        "intent-ttc-mlp-random-h2-s0-v1": "MLP JEPA",
        "intent-ttc-causal-random-h2-s0-v1": "Causal JEPA",
        "intent-ttc-direct-ranker-s1-v1": "Direct ranker",
        "intent-ttc-mlp-random-h1-s0-v1": "MLP, train horizon 1",
        "intent-ttc-mlp-random-h4-s0-v1": "MLP, train horizon 4",
        "intent-ttc-mlp-greedy-h2b4-s0-v1": "MLP, greedy horizon 2",
        "intent-ttc-mlp-greedy-h4b8-s0-v1": "MLP, greedy horizon 4",
    }
    out = {}
    for run, label in labels.items():
        d = load(base / run / "scaling_summary.json")
        out[label] = {}
        for depth in (1, 2, 4, 8):
            row = d["cells"][f"depth{depth}_cap64"]["0"]
            out[label][depth] = {
                "strict": row["metrics"]["latent_planner"]["success"],
                "flops_per_episode": row["compute"]["measured_flops_per_episode"],
            }
    return out


def ood_results() -> dict:
    base = RUNS / "2026-08-03-intent-distractor-matched-length-ood-lise-v4"
    paths = {
        "Sentence LM": base / "intent-length-ood-matched-sentence-lm-s0-v4/length_curve.json",
        "Sentence LM + latent": base / "intent-length-ood-matched-sentence-latent-lm-s0-v4/length_curve.json",
    }
    out = {}
    for name, p in paths.items():
        cells = load(p)["cells"]
        out[name] = {int(k): {"strict": v["0"]["success"], "slack2": v["2"]["success"]} for k, v in cells.items()}
    return out


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def plot_headline(main: dict) -> None:
    order = ["Sentence LM + latent", "Sentence LM", "MLP JEPA", "Token LM", "Causal JEPA"]
    colors = [COLORS["hybrid"], COLORS["sentence"], COLORS["jepa"], COLORS["token"], COLORS["causal"]]
    y = np.arange(len(order)); h = 0.34
    fig, ax = plt.subplots(figsize=(7.1, 3.25))
    ax.barh(y + h / 2, [main[x]["strict_mean"] for x in order], h,
            xerr=[main[x]["strict_sd"] for x in order], color=colors, alpha=.95, label="Strict")
    ax.barh(y - h / 2, [main[x]["slack2_mean"] for x in order], h,
            xerr=[main[x]["slack2_sd"] for x in order], color=colors, alpha=.35, label="Two extra actions")
    ax.set_yticks(y, order); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("Puzzle success rate (mean ± seed SD; five seeds)")
    ax.set_title("JEPA is competitive; the sentence-plus-latent model leads strict success")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=.18)
    save(fig, "headline_models")


def plot_gar(abls: dict, controls: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.6), gridspec_kw={"wspace": .55})
    left = ["Full GAR", "No GAR", "Latent MSE only", "Ranking only", "No counterfactual latent", "GAR horizon 1", "GAR horizon 4"]
    y = np.arange(len(left))
    axes[0].barh(y, [abls[x]["strict_mean"] for x in left],
                 xerr=[abls[x]["strict_sd"] for x in left],
                 color=[COLORS["jepa"] if x == "Full GAR" else COLORS["control"] for x in left])
    axes[0].set_yticks(y, left); axes[0].invert_yaxis(); axes[0].set_xlim(0, .55)
    axes[0].set_xlabel("Strict success (5 seeds)")
    axes[0].set_title("GAR is the decisive objective")
    axes[0].grid(axis="x", alpha=.18)
    right = ["Full GAR", "Detached GAR body", "GAR, no CF latent", "CF latent only", "Direct ranker", "Raw geometry only"]
    y = np.arange(len(right))
    axes[1].barh(y, [controls[x]["strict"] for x in right],
                 color=[COLORS["jepa"], COLORS["causal"], COLORS["control"], COLORS["control"], COLORS["token"], COLORS["bad"]])
    axes[1].set_yticks(y, right); axes[1].invert_yaxis(); axes[1].set_xlim(0, .55)
    axes[1].set_xlabel("Strict success (single seed)")
    axes[1].set_title("Scorer vs. geometry controls")
    axes[1].grid(axis="x", alpha=.18)
    axes[1].text(.02, -.52, "Exploratory: one seed per row", transform=axes[1].transAxes, fontsize=7, color="#555")
    save(fig, "gar_mechanism")


def plot_representations(main: dict, rep: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), gridspec_kw={"wspace": .35})
    names = list(main)
    color = {"MLP JEPA": COLORS["jepa"], "Causal JEPA": COLORS["causal"], "Token LM": COLORS["token"], "Sentence LM": COLORS["sentence"], "Sentence LM + latent": COLORS["hybrid"]}
    for name in names:
        axes[0].scatter(rep[name]["effective_rank_mean"], main[name]["strict_mean"], s=60, color=color[name], edgecolor="white", linewidth=.7, zorder=3)
        axes[0].annotate(name, (rep[name]["effective_rank_mean"], main[name]["strict_mean"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[0].set_xlabel("Effective latent rank")
    axes[0].set_ylabel("Strict planning success")
    axes[0].set_title("Non-collapse is not planning quality")
    axes[0].grid(alpha=.18)
    x=np.arange(len(names)); w=.25
    axes[1].bar(x-w, [rep[n]["operation_balanced_accuracy_mean"] for n in names], w, label="Operation identity")
    axes[1].bar(x, [rep[n]["remaining_steps_r2_mean"] for n in names], w, label="Remaining steps")
    axes[1].bar(x+w, [rep[n]["resolved_count_r2_mean"] for n in names], w, label="Resolved count*")
    axes[1].set_xticks(x, [n.replace("Sentence LM + latent", "Sent.+latent").replace("Sentence LM", "Sent. LM").replace("Causal JEPA", "Causal").replace("MLP JEPA", "MLP") for n in names], rotation=25, ha="right")
    axes[1].set_ylim(0,1); axes[1].set_ylabel("Probe score")
    axes[1].set_title("Different information is linearly accessible")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].text(0, -.38, "*Resolved count equals step index in these traces.", transform=axes[1].transAxes, fontsize=7, color="#555")
    axes[1].grid(axis="y", alpha=.18)
    save(fig, "representation_diagnostics")


def plot_test_time(looped: dict, scaling: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"wspace": .32})
    colors={"Token LM":COLORS["token"],"Sentence LM":COLORS["sentence"],"Sentence LM + latent":COLORS["hybrid"]}
    for name, cells in looped.items():
        xs=sorted(cells); axes[0].plot(xs,[cells[x]["strict"] for x in xs],marker="o",label=name,color=colors[name])
    axes[0].set_xscale("log",base=2); axes[0].set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32])
    axes[0].set_ylim(0,1); axes[0].set_xlabel("Recurrent loops at evaluation")
    axes[0].set_ylabel("Strict success")
    axes[0].set_title("Valid one-seed recurrence pilot")
    axes[0].legend(frameon=False,loc="upper left")
    axes[0].grid(alpha=.18)
    selected=["MLP JEPA","Causal JEPA","Direct ranker","MLP, train horizon 1","MLP, train horizon 4"]
    cols=[COLORS["jepa"],COLORS["causal"],COLORS["token"],"#79B7DB","#194F87"]
    for name,c in zip(selected,cols):
        cells=scaling[name]; xs=sorted(cells); axes[1].plot(xs,[cells[x]["strict"] for x in xs],marker="o",label=name,color=c)
    axes[1].set_xscale("log",base=2); axes[1].set_xticks([1,2,4,8],[1,2,4,8]); axes[1].set_ylim(0,1.03)
    axes[1].set_xlabel("Symbolic future-action depth")
    axes[1].set_title("Invalid: terminal paths leak through length")
    axes[1].axvspan(2.8,8.5,color="#FCE8E6",alpha=.7,zorder=0)
    axes[1].text(3.15,.12,"All systems → 0.97\nProtocol artifact, not JEPA scaling",fontsize=8,color=COLORS["bad"],weight="bold")
    axes[1].legend(frameon=False,fontsize=6.6,loc="center right")
    axes[1].grid(alpha=.18)
    save(fig, "test_time_compute")


def plot_ood(ood: dict) -> None:
    fig, ax=plt.subplots(figsize=(7.0,3.1))
    styles={"Sentence LM":COLORS["sentence"],"Sentence LM + latent":COLORS["hybrid"]}
    for name,cells in ood.items():
        xs=sorted(cells); col=styles[name]
        ax.plot(xs,[cells[x]["strict"] for x in xs],marker="o",color=col,label=f"{name}, strict")
        ax.plot(xs,[cells[x]["slack2"] for x in xs],marker="s",ls="--",color=col,alpha=.65,label=f"{name}, +2")
    ax.axvspan(9.5,11.5,color="#FCE8E6",alpha=.7,label="Beyond training lengths")
    ax.axvline(9.5,color=COLORS["bad"],lw=.8)
    ax.set_xticks([3,5,7,9,10,11]); ax.set_ylim(0,1.03)
    ax.set_xlabel("Required solution length"); ax.set_ylabel("Success rate")
    ax.set_title("Longer problems expose an unresolved generalization gap (one seed)")
    ax.legend(frameon=False,ncol=2,fontsize=7,loc="lower left")
    ax.grid(alpha=.18)
    save(fig,"length_ood")


def main() -> None:
    style()
    data = {
        "main": main_results(),
        "ablations": ablations(),
        "mechanism_controls": mechanism_controls(),
        "representations": representation_results(),
        "looped": looped_results(),
        "candidate_privileged_scaling": scaling_results(),
        "length_ood": ood_results(),
    }
    (OUT / "report_data.json").write_text(json.dumps(data, indent=2) + "\n")
    plot_headline(data["main"])
    plot_gar(data["ablations"], data["mechanism_controls"])
    plot_representations(data["main"], data["representations"])
    plot_test_time(data["looped"], data["candidate_privileged_scaling"])
    plot_ood(data["length_ood"])


if __name__ == "__main__":
    main()
