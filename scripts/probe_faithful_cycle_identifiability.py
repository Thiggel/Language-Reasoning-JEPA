"""Is the faithful LDAD cycle decode action-identifiable at all? (diagnostic)

Companion to ``scripts/probe_faithful_cycle_auc.py`` (feasibility AUC = .475,
chance).  Chance AUC has two very different readings:

  (a) the imagined displacement for candidate ``c`` decodes ``c``'s OWN
      phrase regardless of feasibility (identity passes through the
      predictor; the score carries action identity but no preconditions), or
  (b) the decode is garbage for every candidate (no signal at all).

Protocol: at each visited state (oracle random feasible walk, candidate-
privileged diagnostic), compute the [V, L, vocab] cycle logits for all V
remaining catalogue actions and score the full V x V matrix of (displacement
of c_i, phrase of c_j).  Report the rank of the own phrase (diagonal) split
by the feasibility of c_i.  High top-1 in BOTH groups = reading (a).

Usage:
  .venv/bin/python scripts/probe_faithful_cycle_identifiability.py \
      --ckpt <best.pt> --n-problems 8 --device cpu --out out.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from textjepa.planning.faithful_search import FaithfulPlanner
from textjepa.planning.ldad_decode import delta_logits, phrase_log_probs
from textjepa.data.faithful import FaithfulEnv
from textjepa.utils.checkpoint import build_dataset, load_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-problems", type=int, default=8)
    ap.add_argument("--max-op", type=int, default=21)
    ap.add_argument("--max-edge", type=int, default=28)
    ap.add_argument("--op-lo", type=int, default=3)
    ap.add_argument("--op-hi", type=int, default=21)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, vocab, cfg = load_run(args.ckpt, device=str(device))
    cfg.data.max_op = args.max_op
    cfg.data.max_edge = args.max_edge
    cfg.data.op_range = [args.op_lo, args.op_hi]
    dataset = build_dataset(cfg, vocab, "val", size=args.n_problems)

    planner = FaithfulPlanner(model, vocab, device)
    rng = random.Random(args.seed)
    stats = {True: {"n": 0, "top1": 0, "rr": 0.0},
             False: {"n": 0, "top1": 0, "rr": 0.0}}

    with torch.no_grad():
        for i in range(args.n_problems):
            fp, _ = dataset.problem(i)
            env = FaithfulEnv(fp)
            pt = planner._tokens(fp.prompt_sentences)
            pm = torch.ones(1, pt.shape[1], dtype=torch.bool, device=device)
            step_texts: list[str] = []
            while not env.solved:
                state = planner._state(pt, pm, step_texts)
                catalogue = [q for q in env.fp.action_order
                             if q not in env.resolved]
                if len(catalogue) < 2:
                    break
                feas = set(env.feasible_actions())
                phrases = [env.action_text(q) for q in catalogue]
                token_ids = [vocab.encode(p) for p in phrases]
                codes = model.encode_actions(
                    planner._tokens(phrases).squeeze(0).unsqueeze(1)
                ).squeeze(1)
                logits = delta_logits(model, state, codes,
                                      context="cycle identifiability probe")
                V = len(catalogue)
                # score matrix: row i = displacement of c_i vs every phrase j
                for row in range(V):
                    scores = phrase_log_probs(
                        logits[row].unsqueeze(0).expand(V, -1, -1), token_ids
                    )
                    rank = int((scores > scores[row]).sum().item()) + 1
                    group = stats[catalogue[row] in feas]
                    group["n"] += 1
                    group["top1"] += int(rank == 1)
                    group["rr"] += 1.0 / rank
                q = rng.choice(sorted(feas))
                step_texts.append(env.step(q))

    report = {
        "ckpt": args.ckpt,
        "n_problems": args.n_problems,
        "evidence_label": "candidate-privileged oracle-labeled diagnostic",
    }
    for feasible, g in stats.items():
        key = "feasible" if feasible else "infeasible"
        report[key] = {
            "n": g["n"],
            "own_phrase_top1": g["top1"] / g["n"] if g["n"] else None,
            "own_phrase_mrr": g["rr"] / g["n"] if g["n"] else None,
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
