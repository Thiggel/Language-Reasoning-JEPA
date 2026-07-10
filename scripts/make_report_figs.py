"""Generate the report figures (PDF) from run artifacts.

    .venv2/bin/python scripts/make_report_figs.py

Each figure makes exactly one claim; colors follow the entity (a run keeps
its color everywhere), assigned from the validated categorical palette.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path("runs")
OUT = Path("reports/figs")

# fixed entity -> color (validated palette, light mode)
C = {
    "random": "#8a8984",
    "disc_no_delta": "#eda100",
    "disc_base": "#2a78d6",
    "disc_chunkpred": "#1baf7a",
    "disc_combo": "#008300",
    "disc_valgrad": "#4a3aa7",
    "disc_mono_hi": "#b0457b",
    "disc_rank_k2": "#c2361b",
    "edit_base": "#2a78d6",
    "edit_valgrad": "#4a3aa7",
    "edit_anchor": "#1baf7a",
    "oracle": "#0b0b0b",
    "random_ctrl": "#c3c2b7",
    # the two energies (used wherever value-head vs goal-distance is compared)
    "energy_value": "#4a3aa7",
    "energy_goal": "#1baf7a",
}
LABELS = {
    "random": "random policy",
    "disc_no_delta": "no Delta-JEPA",
    "disc_base": "base JEPA",
    "disc_chunkpred": "+ frozen anchor",
    "disc_combo": "+ anchor + value-grad",
    "disc_valgrad": "+ value-grad",
    "disc_mono_hi": "+ monotonicity",
    "disc_rank_k2": "+ ranking (K=2)",
    "edit_base": "base",
    "edit_valgrad": "+ value-grad",
    "edit_anchor": "+ anchor + value-grad",
    "edit_mono": "+ monotonicity",
    "edit_geo": "+ straighten + mono",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e8e7e2",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
})


def plan(run: str, slack=0, look=1, energy="") -> dict | None:
    suffix = f"_{energy}" if energy else ""
    f = RUNS / run / f"plan_slack{slack}_look{look}{suffix}.json"
    return json.loads(f.read_text()) if f.exists() else None


def planner_success(run: str, slack=0, look=1, energy="") -> float | None:
    d = plan(run, slack, look, energy)
    if d is None:
        return None
    key = next(k for k in d if k.startswith("latent_planner"))
    return d[key]["success"]


def bar_with_labels(ax, names, values, colors):
    x = range(len(names))
    bars = ax.bar(x, values, width=0.62, color=colors, zorder=2)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5, color="#0b0b0b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylim(0, 1.02)


def fig_planning():
    order = ["random", "disc_no_delta", "disc_base", "disc_chunkpred",
             "disc_combo", "disc_valgrad", "disc_mono_hi", "disc_rank_k2"]
    vals, names, cols = [], [], []
    rnd = plan("disc_base")["random_policy"]["success"]
    for r in order:
        v = rnd if r == "random" else planner_success(r)
        if v is None:
            continue
        vals.append(v)
        names.append(LABELS[r])
        cols.append(C[r])
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    bar_with_labels(ax, names, vals, cols)
    ax.axhline(1.0, color=C["oracle"], lw=1, ls="--")
    ax.text(len(names) - 0.45, 0.965, "oracle", fontsize=8, color=C["oracle"],
            ha="right")
    ax.set_ylabel("success @ optimal budget")
    fig.tight_layout()
    fig.savefig(OUT / "planning_success.pdf")


def fig_lookahead():
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for r in ["disc_base", "disc_chunkpred", "disc_combo", "disc_valgrad"]:
        xs, ys = [], []
        for look in (1, 2):
            v = planner_success(r, look=look)
            if v is not None:
                xs.append(look)
                ys.append(v)
        if len(xs) >= 1:
            ax.plot(xs, ys, marker="o", ms=5, lw=2, color=C[r], zorder=3)
            ax.text(xs[-1] + 0.05, ys[-1], LABELS[r], fontsize=8, color=C[r],
                    va="center")
    rnd = plan("disc_base")["random_policy"]["success"]
    ax.axhline(rnd, color=C["random"], lw=1.2, ls=":")
    ax.text(1.0, rnd + 0.02, "random", fontsize=8, color=C["random"])
    ax.set_xticks([1, 2])
    ax.set_xlim(0.85, 2.8)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("planner lookahead (latent search depth)")
    ax.set_ylabel("success @ optimal budget")
    fig.tight_layout()
    fig.savefig(OUT / "lookahead.pdf")


def fig_collusion():
    fig, ax = plt.subplots(figsize=(3.9, 2.6))
    for r in ["disc_base", "disc_chunkpred"]:
        m = pd.read_csv(RUNS / r / "metrics.csv")
        d = m.dropna(subset=["val/probe_value_state"])
        epochs = range(len(d))
        ax.plot(list(epochs), d["val/probe_value_state"], lw=2, color=C[r],
                marker="o", ms=3, zorder=3)
        ax.text(len(d) - 1 + 0.3, d["val/probe_value_state"].iloc[-1],
                LABELS[r], fontsize=8, color=C[r], va="center")
    ax.axhline(0.87, color=C["random_ctrl"], lw=1.5, ls="--")
    ax.text(0.2, 0.885, "random-init encoder control", fontsize=8,
            color="#52514e")
    ax.axhline(1 / 23, color=C["random"], lw=1, ls=":")
    ax.text(0.2, 1 / 23 + 0.015, "chance", fontsize=8, color=C["random"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("value decodability from state $s_t$")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.4, 26)
    fig.tight_layout()
    fig.savefig(OUT / "collusion.pdf")


def fig_arithmetic():
    feats = ["value_from_state", "value_from_pred", "value_from_rollout"]
    ticks = ["from $s_t$\n(encoded)", "from $F(s_{t-1},a_t)$\n(1-step pred.)",
             "from rollout\n($t$-step pred.)"]
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    width = 0.27
    for i, r in enumerate(["disc_base", "disc_chunkpred"]):
        df = pd.read_csv(RUNS / r / "probe_results.csv").set_index("task")
        vals = [df.loc[f, "acc_trained"] for f in feats]
        xs = [j + (i - 0.5) * width for j in range(len(feats))]
        ax.bar(xs, vals, width=width * 0.92, color=C[r], zorder=3,
               label=LABELS[r])
    df = pd.read_csv(RUNS / "disc_base" / "probe_results.csv").set_index("task")
    ctrl = [df.loc[f, "acc_random_enc"] for f in feats]
    for j, v in enumerate(ctrl):
        ax.plot([j - 0.42, j + 0.42], [v, v], color="#0b0b0b", lw=1.2, ls="--",
                zorder=4)
    ax.plot([], [], color="#0b0b0b", lw=1.2, ls="--",
            label="random-encoder control")
    ax.axhline(1 / 23, color=C["random"], lw=1, ls=":")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_ylabel("value decodability")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "arithmetic.pdf")


def fig_energy():
    """Value-head energy vs oracle goal-distance energy (if E1 done)."""
    rows = []
    for r in ["disc_base", "disc_chunkpred", "disc_valgrad"]:
        v = planner_success(r)
        o = planner_success(r, energy="oracle_goal")
        if v is not None and o is not None:
            rows.append((r, v, o))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    width = 0.32
    for i, (key, label, hatch) in enumerate(
        [(1, "learned value head", None), (2, "oracle goal distance", "//")]
    ):
        xs = [j + (i - 0.5) * width for j in range(len(rows))]
        vals = [row[key] for row in rows]
        cols = [C[row[0]] for row in rows]
        bars = ax.bar(xs, vals, width=width * 0.9, color=cols, zorder=3,
                      hatch=hatch, edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=7.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([LABELS[r[0]] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("success @ optimal budget")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#8a8984", label="learned value head"),
        Patch(facecolor="#8a8984", hatch="//", edgecolor="white",
              label="oracle goal distance"),
    ], fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "energy.pdf")


def fig_edit_planning():
    """Claim: on the edit track the raw buffer-encoder geometry (goal
    distance) beats every learned value head."""
    order = ["edit_base", "edit_valgrad", "edit_anchor"]
    rows = [
        (r, planner_success(r), planner_success(r, energy="oracle_goal"))
        for r in order
        if planner_success(r) is not None
    ]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    width = 0.32
    for i, key in enumerate((1, 2)):
        xs = [j + (i - 0.5) * width for j in range(len(rows))]
        vals = [row[key] if row[key] is not None else 0 for row in rows]
        col = C["energy_value"] if key == 1 else C["energy_goal"]
        bars = ax.bar(xs, vals, width=width * 0.9, color=col, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=7.5)
    rnd = plan("edit_base")["random_policy"]["success"]
    ax.axhline(rnd, color=C["random"], lw=1.2, ls=":")
    ax.text(len(rows) - 0.55, rnd + 0.02, "random", fontsize=8,
            color=C["random"])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([LABELS[r[0]] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("perfect draft @ optimal budget")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C["energy_value"], label="learned value head"),
        Patch(facecolor=C["energy_goal"], label="raw goal distance"),
    ], fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "edit_planning.pdf")


STRAIGHT_SWEEP = [
    (0.0, "disc_combo"),
    (0.02, "disc_straight_lo"),
    (0.05, "disc_straight_mid"),
    (0.1, "disc_straight"),
]


def fig_straighten():
    """Claim: temporal straightening buys raw-geometry planning at the
    cost of value-head planning — one dose-response curve per energy."""
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    for energy, col, label in (
        ("", C["energy_value"], "value-head energy"),
        ("oracle_goal", C["energy_goal"], "goal-distance energy"),
    ):
        for slack, ls in ((0, "--"), (2, "-")):
            xs, ys = [], []
            for lam, run in STRAIGHT_SWEEP:
                v = planner_success(run, slack=slack, energy=energy)
                if v is not None:
                    xs.append(lam)
                    ys.append(v)
            ax.plot(xs, ys, marker="o", ms=4, lw=2 if slack else 1.4,
                    ls=ls, color=col, zorder=3,
                    label=f"{label} (slack {slack})")
    ax.set_xticks([lam for lam, _ in STRAIGHT_SWEEP])
    ax.set_xlabel(r"straightening weight $\lambda_{\mathrm{curv}}$")
    ax.set_ylabel("planning success")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "straighten_tradeoff.pdf")


def _audit(run: str) -> dict | None:
    f = RUNS / run / "counterfactual_audit.json"
    return json.loads(f.read_text()) if f.exists() else None


def fig_audit():
    """Claim: LDAD grounds counterfactual transitions — matching of F(s,a)
    to true next states collapses without it; explicit ranking restores it."""
    order = ["disc_value_only", "disc_no_delta", "disc_base", "disc_combo",
             "disc_rank_k4_nodelta", "disc_rank_k4"]
    labels = {"disc_value_only": "value only", "disc_no_delta": "no LDAD",
              "disc_base": "base JEPA", "disc_combo": "combo",
              "disc_rank_k4_nodelta": "rank, no LDAD", "disc_rank_k4": "rank + LDAD"}
    rows = [(r, _audit(r)) for r in order]
    rows = [(r, d) for r, d in rows if d is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    width = 0.36
    for i, (key, label, col) in enumerate([
        ("match", "NN matching of $F(s,a)$", "#2a78d6"),
        ("tau_value", r"ranking Kendall $\tau$ (value energy)", "#eda100"),
    ]):
        xs = [j + (i - 0.5) * width for j in range(len(rows))]
        vals = [d[key] for _, d in rows]
        bars = ax.bar(xs, vals, width=width * 0.9, color=col, zorder=3, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=7)
    chance = sum(d["chance"] for _, d in rows) / len(rows)
    ax.axhline(chance, color=C["random"], lw=1.2, ls=":")
    ax.text(0.02, chance + 0.02, "matching chance", fontsize=7.5, color=C["random"])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([labels[r] for r, _ in rows], fontsize=7.5, rotation=12,
                       ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("counterfactual accuracy")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "audit.pdf")


def fig_datascale():
    """Claim: cross-problem diversity is the counterfactual fuel; ranking
    supervision partially substitutes for it."""
    sizes = [(10, "disc_size10k", "disc_size10k_rank"),
             (30, "disc_size30k", None),
             (100, "disc_combo", "disc_rank_k2")]
    fig, ax = plt.subplots(figsize=(3.9, 2.6))
    for idx, (label, col) in enumerate([("combo", C["disc_combo"]),
                                        ("+ ranking", C["disc_rank_k2"])]):
        xs, ys = [], []
        for size, base_run, rank_run in sizes:
            run = base_run if idx == 0 else rank_run
            v = planner_success(run) if run else None
            if v is not None:
                xs.append(size)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, marker="o", ms=5, lw=2, color=col, zorder=3)
            ax.text(xs[-1], ys[-1] + 0.04, label, fontsize=8, color=col,
                    ha="right")
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 100])
    ax.set_xticklabels(["10k", "30k", "100k"])
    ax.set_xlabel("unique training problems")
    ax.set_ylabel("success @ optimal budget")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "datascale.pdf")


def fig_memory():
    """Claim: the discourse state is a recency-weighted memory, not a full
    symbol table — value decodability decays with lag."""
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for r in ["disc_base", "disc_combo"]:
        f = RUNS / r / "probe_v2.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).set_index("task")
        lags, accs = [0], [df.loc["value_from_state", "acc_trained"]]
        for lag in (1, 2, 3):
            t = f"value_prev{lag}_from_state"
            if t in df.index:
                lags.append(lag)
                accs.append(df.loc[t, "acc_trained"])
        ax.plot(lags, accs, marker="o", ms=5, lw=2, color=C[r], zorder=3)
        ax.text(lags[-1] + 0.07, accs[-1], LABELS[r], fontsize=8, color=C[r],
                va="center")
    ax.axhline(1 / 23, color=C["random"], lw=1, ls=":")
    ax.text(0.0, 1 / 23 + 0.015, "chance", fontsize=8, color=C["random"])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlim(-0.2, 3.9)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("lag (steps since the value was established)")
    ax.set_ylabel("value decodability from $s_t$")
    fig.tight_layout()
    fig.savefig(OUT / "memory.pdf")


def fig_emergence():
    """Claim: answer emergence is a step function — near-chance until the
    query resolves, then snaps to 1.0 (no anticipatory computation); the
    base model cannot even hold it at the terminal step (collusion)."""
    files = [("disc_combo", C["disc_combo"], "-"),
             ("disc_base", C["disc_base"], "--")]
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    plotted = False
    for run, col, ls in files:
        f = RUNS / run / "emergence.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        rem = sorted((int(k) for k in d), reverse=True)
        for key, lw, alpha, label in (("linear", 2.0, 1.0, LABELS[run]),
                                      ("mlp", 1.2, 0.55, None)):
            ax.plot(rem, [d[str(r)][key] for r in rem], marker="o", ms=4,
                    lw=lw, ls=ls, color=col, alpha=alpha, zorder=3,
                    label=f"{label} (linear)" if label else None)
        plotted = True
    if not plotted:
        return
    ax.plot([], [], color="#52514e", lw=1.2, alpha=0.55, label="(thin: MLP probe)")
    ax.axhline(1 / 23, color=C["random"], lw=1, ls=":")
    ax.text(5.0, 1 / 23 + 0.02, "chance", fontsize=8, color=C["random"])
    ax.invert_xaxis()  # progress runs left->right (remaining decreases)
    ax.set_xticks([5, 4, 3, 2, 1, 0])
    ax.set_xlabel("necessary steps remaining (progress $\\rightarrow$)")
    ax.set_ylabel("answer decodability from $s_t$")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "emergence.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_planning()
    fig_lookahead()
    fig_collusion()
    fig_arithmetic()
    fig_energy()
    fig_edit_planning()
    fig_straighten()
    fig_audit()
    fig_datascale()
    fig_memory()
    fig_emergence()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
