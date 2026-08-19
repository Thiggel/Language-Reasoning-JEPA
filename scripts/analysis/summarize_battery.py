"""Render the battery JSONs as the markdown tables used in the report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(x, n=3):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return "--"


def geometry_table(path: Path, space: str = "action_ctx",
                   dist: str = "cosine") -> str:
    d = json.loads(path.read_text())
    lines = [f"### {path.name}  (space={space}, distance={dist})",
             f"contexts={d.get('n_contexts')} pairs={d.get('n_pairs')} "
             f"commutation_cases={d.get('n_commutation_cases')}", ""]
    sb = d.get("surface_edit_distance_baseline", {})
    if sb.get("same_vs_all_different"):
        s = sb["same_vs_all_different"]
        lines += [f"surface token-edit baseline: same={f(s['same_mean'],2)} "
                  f"diff={f(s['diff_mean'],2)} ratio={f(s['separation_ratio'])} "
                  f"AUC={f(s['auc_diff_vs_same'])}", ""]
    subkinds: list[str] = []
    for m in d["models"].values():
        pg = m.get("pair_geometry", {}).get(space, {}).get(dist, {})
        for k in pg.get("per_subkind", {}):
            if k not in subkinds:
                subkinds.append(k)
    if subkinds:
        lines.append("| model | " + " | ".join(subkinds)
                     + " | ratio | AUC same-vs-diff |")
        lines.append("|" + "---|" * (len(subkinds) + 3))
        for name, m in d["models"].items():
            pg = m.get("pair_geometry", {}).get(space, {}).get(dist, {})
            if not pg:
                continue
            cells = [f(pg["per_subkind"].get(k, {}).get("mean"), 4)
                     for k in subkinds]
            sv = pg.get("same_vs_all_different", {})
            lines.append(f"| {name} | " + " | ".join(cells)
                         + f" | {f(sv.get('separation_ratio'),2)} "
                           f"| {f(sv.get('auc_diff_vs_same'))} |")
        lines.append("")
    rows = [(n, m["state_order_invariance"][dist])
            for n, m in d["models"].items() if "state_order_invariance" in m]
    if rows:
        lines += ["state order-invariance (A-then-B vs B-then-A = same "
                  "consequence; vs A-then-C = different)", "",
                  "| model | same | diff | ratio | AUC |", "|---|---|---|---|---|"]
        for n, r in rows:
            lines.append(f"| {n} | {f(r['same_mean'],4)} | {f(r['diff_mean'],4)} "
                         f"| {f(r['separation_ratio'],2)} "
                         f"| {f(r['auc_diff_vs_same'])} |")
        lines.append("")
    return "\n".join(lines)


def normalized_table(path: Path, reference: str, space: str = "action_ctx",
                     dist: str = "cosine") -> str:
    """Distances rescaled by a within-model reference distance.

    Raw distances are not comparable across encoders: a nearly-collapsed
    random-init encoder has tiny distances everywhere, which inflates every
    ratio.  Dividing each pair family by the SAME model's distance for a
    generic different-consequence pair (``reference``) removes that scale and
    asks the question that actually matters:

      * ``negation_salience`` -- how far apart does a ONE-TOKEN meaning flip
        push two phrases, as a fraction of the distance between two wholly
        different actions?  Higher = the model treats a small surface change
        with a big consequence as a big change.
      * ``paraphrase_leakage`` -- how far apart does a same-consequence
        rewording push them, on the same scale?  Lower is better; a value
        near or above 1 means the encoder is tracking words, not consequences.
    """
    d = json.loads(path.read_text())
    rows = []
    for name, m in d["models"].items():
        pg = m.get("pair_geometry", {}).get(space, {}).get(dist, {})
        sub = pg.get("per_subkind", {})
        ref = sub.get(reference, {}).get("mean")
        if not ref:
            continue
        entry = {"model": name}
        for k, v in sub.items():
            if k == reference:
                continue
            entry[k] = v["mean"] / ref
        rows.append(entry)
    if not rows:
        return ""
    keys = [k for k in rows[0] if k != "model"]
    lines = [f"### {path.name} -- scale-normalized (each family / same model's "
             f"`{reference}` distance; space={space}, {dist})", "",
             "| model | " + " | ".join(keys) + " |",
             "|" + "---|" * (len(keys) + 1)]
    for r in rows:
        lines.append(f"| {r['model']} | "
                     + " | ".join(f(r.get(k), 4) for k in keys) + " |")
    lines.append("")
    return "\n".join(lines)


def probe_table(path: Path, head: str = "mlp") -> str:
    d = json.loads(path.read_text())
    lines = [f"### {path.name}  (probe head = {head})", ""]
    tasks = ["resolved", "feasible", "remaining_steps_DIAGNOSTIC",
             "step_value", "operator_from_displacement"]
    metric = {"resolved": "auc", "feasible": "auc",
              "remaining_steps_DIAGNOSTIC": "r2", "step_value": "r2",
              "operator_from_displacement": "acc"}
    lines.append("| model | " + " | ".join(
        f"{t} ({metric[t]})" for t in tasks) + " |")
    lines.append("|" + "---|" * (len(tasks) + 1))
    for name, m in d["models"].items():
        cells = []
        for t in tasks:
            b = m["probes"].get(t, {}).get(head, {})
            cells.append(f(b.get(metric[t])))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    for t in ("resolved", "feasible"):
        depths: list[int] = []
        for m in d["models"].values():
            for k in m["probes"][t][head].get("by_depth", {}):
                if int(k) not in depths:
                    depths.append(int(k))
        depths.sort()
        lines += [f"per-depth AUC, {t}", "",
                  "| model | " + " | ".join(f"d{x}" for x in depths) + " |",
                  "|" + "---|" * (len(depths) + 1)]
        for name, m in d["models"].items():
            bd = m["probes"][t][head].get("by_depth", {})
            lines.append(f"| {name} | " + " | ".join(
                f(bd.get(str(x), {}).get("auc")) for x in depths) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", action="append", default=[])
    ap.add_argument("--probes", action="append", default=[])
    ap.add_argument("--space", default="action_ctx")
    ap.add_argument("--dist", default="cosine")
    ap.add_argument("--normalize-by", default=None,
                    help="subkind used as the within-model scale reference")
    args = ap.parse_args()
    for p in args.geometry:
        print(geometry_table(Path(p), args.space, args.dist))
        if args.normalize_by:
            print(normalized_table(Path(p), args.normalize_by, args.space,
                                   args.dist))
    for p in args.probes:
        print(probe_table(Path(p)))


if __name__ == "__main__":
    main()
