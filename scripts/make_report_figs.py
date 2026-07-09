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
    "edit_base": "#2a78d6",
    "edit_valgrad": "#4a3aa7",
    "edit_anchor": "#1baf7a",
    "oracle": "#0b0b0b",
    "random_ctrl": "#c3c2b7",
}
LABELS = {
    "random": "random policy",
    "disc_no_delta": "no Delta-JEPA",
    "disc_base": "base JEPA",
    "disc_chunkpred": "+ frozen anchor",
    "disc_combo": "+ anchor + value-grad",
    "disc_valgrad": "+ value-grad",
    "edit_base": "base",
    "edit_valgrad": "+ value-grad",
    "edit_anchor": "+ anchor + value-grad",
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
             "disc_combo", "disc_valgrad"]
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
    order = ["edit_base", "edit_valgrad", "edit_anchor"]
    names, cols = ["random policy"], [C["random"]]
    vals = [plan("edit_base")["random_policy"]["success"]]
    for r in order:
        v = planner_success(r)
        if v is None:
            continue
        names.append(LABELS[r])
        vals.append(v)
        cols.append(C[r])
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    bar_with_labels(ax, names, vals, cols)
    ax.axhline(1.0, color=C["oracle"], lw=1, ls="--")
    ax.text(len(names) - 0.45, 0.965, "oracle", fontsize=8, ha="right")
    ax.set_ylabel("perfect draft @ optimal budget")
    fig.tight_layout()
    fig.savefig(OUT / "edit_planning.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_planning()
    fig_lookahead()
    fig_collusion()
    fig_arithmetic()
    fig_energy()
    fig_edit_planning()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
