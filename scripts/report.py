"""Aggregate planning + probe results across runs into markdown tables.

    python scripts/report.py [runs_dir] [out_md]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def planning_table(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs_dir.glob("*/plan_slack*_look*.json")):
        data = json.loads(f.read_text())
        parts = f.stem.replace("plan_slack", "").split("_look")
        for policy, m in data.items():
            rows.append(
                {
                    "run": f.parent.name,
                    "slack": int(parts[0]),
                    "lookahead": int(parts[1]),
                    "policy": policy,
                    "success": m["success"],
                    "distractor_rate": m.get("distractor_rate"),
                }
            )
    return pd.DataFrame(rows)


def probe_table(runs_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(runs_dir.glob("*/probe_results.csv")):
        df = pd.read_csv(f)
        df.insert(0, "run", f.parent.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else runs_dir / "report.md"

    plans = planning_table(runs_dir)
    probes = probe_table(runs_dir)

    lines = ["# TextJEPA results\n", "## Planning success\n"]
    if not plans.empty:
        piv = plans[plans.policy == "latent_planner"].pivot_table(
            index="run", columns=["slack", "lookahead"], values="success"
        )
        base = plans[plans.policy == "random_policy"].pivot_table(
            index="run", columns=["slack", "lookahead"], values="success"
        )
        lines += [piv.round(3).to_markdown(), "\n### Random-policy baseline\n",
                  base.round(3).to_markdown(), "\n"]
    if not probes.empty:
        lines.append("\n## Probes (trained vs random-encoder control)\n")
        cols = [c for c in ("run", "task", "acc_trained", "acc_random_enc", "majority")
                if c in probes.columns]
        lines.append(probes[cols].round(3).to_markdown(index=False))
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    if not plans.empty:
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
