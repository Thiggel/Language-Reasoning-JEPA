"""Build the compute-matched baseline table from a test-time-compute round.

    .venv2/bin/python scripts/summarize_testtime_compute.py <round_dir>

Prints one markdown table per checkpoint: success and steps-used against the
compute parameter (N samples / K loops) AND against generated tokens per
episode, plus the JEPA planner rows from ``jepa-compute-axis/`` on the same
backbone-token-positions axis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def rows_from_lm(path: Path):
    d = json.loads(path.read_text())
    for label, m in d.items():
        if label == "protocol" or not isinstance(m, dict):
            continue
        c = m["compute"]["per_episode"]
        yield {
            "row": label,
            "success": m["success_rate"],
            "answer": m["success_answer_rate"],
            "oracle_pass@n": m["oracle_pass_at_n"],
            "self_cons": m.get("self_consistency_success_rate", float("nan")),
            "steps/nec": m["solved_steps_over_necessary_mean"],
            "gen_tok/ep": c["generated_tokens_per_episode"],
            "tokpos/ep": c["backbone_token_positions_per_episode"],
        }


def rows_from_jepa(path: Path):
    d = json.loads(path.read_text())
    m, c = d["latent_planner"], d["compute"]["per_episode"]
    yield {
        "row": f"JEPA {d['protocol']['interface']} d{d['protocol']['lookahead']}",
        "success": m["success"], "answer": m.get("success_answer_rate",
                                                 float("nan")),
        "oracle_pass@n": float("nan"),
        "self_cons": float("nan"),
        "steps/nec": m["solved_steps_over_necessary_mean"],
        "gen_tok/ep": c["generated_tokens_per_episode"],
        "tokpos/ep": c["backbone_token_positions_per_episode"],
    }


def table(rows):
    cols = ["row", "success", "answer", "oracle_pass@n", "self_cons",
            "steps/nec", "gen_tok/ep", "tokpos/ep"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        cells = [str(r["row"])] + [
            ("%.3f" % r[c]) if r[c] == r[c] and c not in
            ("gen_tok/ep", "tokpos/ep") else
            ("%.0f" % r[c] if r[c] == r[c] else "-")
            for c in cols[1:]
        ]
        print("| " + " | ".join(cells) + " |")
    print()


def main() -> None:
    root = Path(sys.argv[1])
    for cell in sorted(root.glob("ttc-*")):
        for js in sorted(cell.glob("testtime_*.json")):
            print(f"### {cell.name} / {js.stem}")
            table(list(rows_from_lm(js)))
    jd = root / "jepa-compute-axis"
    if jd.exists():
        rows = []
        for js in sorted(jd.glob("flat_*.json")):
            rows.extend(rows_from_jepa(js))
        if rows:
            print("### JEPA planner on the same compute axis")
            table(rows)


if __name__ == "__main__":
    main()
