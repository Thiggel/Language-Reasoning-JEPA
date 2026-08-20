"""Re-score completed plan_flat.py runs at tighter step budgets.

Valid because no policy reads its own budget: the episode at cap c is a strict
prefix of the episode the run actually executed, so success at cap c is exactly
``solved_at <= ceil(c * necessary)``.  This lets every plan JSON already on
disk be reported at any budget without rerunning anything.

Context (2026-08-20 interface audit): the default cap of 4x necessary steps is
what made the reference policies look competent -- random scores .595 at cap 4
and .000 at cap 1.0 on full_catalogue, so our margin over guessing was
understated by roughly 2x.  Reference-policy per-episode records come from
``episodes_random`` / ``episodes_first_feasible`` (commit 043b34f); files
written before that carry only the planner's own episodes and are reported
without reference columns.

Answer-emission criterion (2026-08-21): plan JSONs written with answer
emission on carry ``answer_correct`` per episode (the model's own generated
final outcome sentence, properly terminated, states the true answer).  The
budget-rescoring rule under it is::

    success_answer at cap c  =  (solved_at <= ceil(c * necessary))
                                AND answer_correct is True

This is exact, not an approximation: the answer is emitted AT the solving
step, so whenever ``solved_at`` lies inside cap c the emission happened
inside cap c as well, and an episode that only hit the runaway cap has
``solved_at = None`` and never counts at any cap ("terminated by the model's
own progress, not by budget exhaustion" is therefore implied).  Files
predating the criterion have ``answer_correct`` absent/None everywhere and
report answer columns as ``--``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CAPS = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
ARMS = (("episodes", "model"), ("episodes_random", "random"),
        ("episodes_first_feasible", "first"))


def success_at(episodes: list, cap: float) -> float | None:
    if not episodes:
        return None
    ok = sum(1 for e in episodes
             if e.get("solved_at") is not None
             and e["solved_at"] <= math.ceil(cap * e["necessary"]))
    return ok / len(episodes)


def success_answer_at(episodes: list, cap: float) -> float | None:
    """Answer-emission success at cap ``c`` (see module docstring for the
    exact rule and why prefix-rescoring stays valid under it)."""
    if not episodes or not any(e.get("answer_correct") is not None for e in episodes):
        return None
    ok = sum(1 for e in episodes
             if e.get("solved_at") is not None
             and e["solved_at"] <= math.ceil(cap * e["necessary"])
             and e.get("answer_correct") is True)
    return ok / len(episodes)


def rescore(path: Path, caps=CAPS) -> dict:
    d = json.loads(path.read_text())
    proto = d.get("protocol", {})
    out = {"file": str(path), "interface": proto.get("interface"),
           "lookahead": proto.get("lookahead"), "cap_mult_run": proto.get("cap_mult"),
           "n_episodes": len(d.get("episodes", [])), "caps": {}}
    for cap in caps:
        row = {label: success_at(d.get(key, []), cap) for key, label in ARMS}
        for key, label in ARMS:
            row[f"{label}_answer"] = success_answer_at(d.get(key, []), cap)
        if row["model"] is not None and row["random"] is not None:
            row["gap"] = round(row["model"] - row["random"], 4)
        if row["model_answer"] is not None and row["random_answer"] is not None:
            row["gap_answer"] = round(row["model_answer"] - row["random_answer"], 4)
        out["caps"][str(cap)] = {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in row.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    results = []
    for p in args.paths:
        try:
            r = rescore(p)
        except Exception as exc:  # a malformed or non-plan JSON must not stop the sweep
            print(f"!! {p}: {type(exc).__name__}: {exc}")
            continue
        results.append(r)
        print(f"== {p.name}  {r['interface']} d{r['lookahead']}  "
              f"n={r['n_episodes']}  (run at cap {r['cap_mult_run']})")
        for cap, row in r["caps"].items():
            fmt = lambda v: "  --" if v is None else f"{v:.3f}"
            gap = "" if row.get("gap") is None else f"  gap={row['gap']:+.3f}"
            print(f"   cap {cap:>5}: model={fmt(row['model'])} random={fmt(row['random'])} "
                  f"first={fmt(row['first'])}{gap}")
            if row.get("model_answer") is not None:
                gap_a = ("" if row.get("gap_answer") is None
                         else f"  gap={row['gap_answer']:+.3f}")
                print(f"      answer: model={fmt(row['model_answer'])} "
                      f"random={fmt(row['random_answer'])} "
                      f"first={fmt(row['first_answer'])}{gap_a}")
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
