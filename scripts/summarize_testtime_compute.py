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
            # Self-consistency is an ANSWER metric: majority vote over the
            # final answers the model wrote.  Attempts that never reached the
            # goal write no answer and therefore abstain, which is why
            # ``self_consistency_success_rate`` in the JSON tracks pass@N and
            # must NOT be quoted as a success rate -- use this one.
            "self_cons_ans": m.get("self_consistency_answer_accuracy",
                                   float("nan")),
            "rerank_sum": m.get("rerank_sumlogp_success_rate", float("nan")),
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
        "self_cons_ans": float("nan"),
        "rerank_sum": float("nan"),
        "steps/nec": m["solved_steps_over_necessary_mean"],
        "gen_tok/ep": c["generated_tokens_per_episode"],
        "tokpos/ep": c["backbone_token_positions_per_episode"],
    }


def table(rows):
    cols = ["row", "success", "answer", "rerank_sum", "self_cons_ans",
            "oracle_pass@n", "steps/nec", "gen_tok/ep", "tokpos/ep"]
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
