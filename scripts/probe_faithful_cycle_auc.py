"""LDAD cycle-consistency feasibility AUC on FAITHFUL iGSM (diagnostic).

The stylized track's menu-free story rests on the cycle score separating
feasible from infeasible actions (AUC ~.94 there).  On faithful iGSM the
cycle-ranked planners fail (invalid rate above random), so this probe asks
the direct question: does the cycle score carry ANY feasibility signal on a
faithful checkpoint, and does it degrade off the training distribution?

Protocol: sample problems, walk a random FEASIBLE trajectory (oracle-driven
-- this is a labeled candidate-privileged diagnostic, not a planning
result), and at every visited state score every catalogue action with the
shared LDAD cycle path (delta_logits + phrase_log_probs).  Labels come from
env.feasible_actions().  Report ROC AUC pooled and per-state mean.

Usage:
  .venv/bin/python scripts/probe_faithful_cycle_auc.py --ckpt <best.pt> \
      --n-problems 32 --max-op 21 --max-edge 28 --op-lo 3 --op-hi 21 \
      --device cpu --out out.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from textjepa.planning.faithful_search import FaithfulPlanner
from textjepa.planning.ldad_decode import delta_logits, phrase_log_probs
from textjepa.data.faithful import FaithfulDataset, FaithfulEnv
from textjepa.utils.checkpoint import build_dataset, load_run


def pooled_auc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        mean_rank = (i + j - 1) / 2 + 1
        rank_sum += mean_rank * sum(lab for _, lab in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-problems", type=int, default=32)
    ap.add_argument("--max-op", type=int, default=21)
    ap.add_argument("--max-edge", type=int, default=28)
    ap.add_argument("--op-lo", type=int, default=3)
    ap.add_argument("--op-hi", type=int, default=21)
    ap.add_argument("--nec-lo", type=int, default=0)
    ap.add_argument("--nec-hi", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, vocab, cfg = load_run(args.ckpt, device=str(device))

    cfg.data.max_op = args.max_op
    cfg.data.max_edge = args.max_edge
    cfg.data.op_range = [args.op_lo, args.op_hi]
    if args.nec_hi:
        cfg.data.necessary_range = [args.nec_lo, args.nec_hi]
    dataset = build_dataset(cfg, vocab, args.split, size=args.n_problems)

    planner = FaithfulPlanner(model, vocab, device)
    rng = random.Random(args.seed)
    all_scores: list[float] = []
    all_labels: list[int] = []
    per_state_auc: list[float] = []

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
                codes = model.encode_actions(
                    planner._tokens(phrases).squeeze(0).unsqueeze(1)
                ).squeeze(1)
                scores = phrase_log_probs(
                    delta_logits(model, state, codes,
                                 context="faithful cycle AUC probe"),
                    [vocab.encode(p) for p in phrases],
                ).tolist()
                labels = [int(q in feas) for q in catalogue]
                all_scores += scores
                all_labels += labels
                auc = pooled_auc(scores, labels)
                if auc == auc:
                    per_state_auc.append(auc)
                q = rng.choice(sorted(feas))
                step_texts.append(env.step(q))

    report = {
        "ckpt": args.ckpt,
        "caps": {"max_op": args.max_op, "max_edge": args.max_edge,
                 "op_range": [args.op_lo, args.op_hi],
                 "necessary_range": ([args.nec_lo, args.nec_hi]
                                     if args.nec_hi else None)},
        "n_problems": args.n_problems,
        "n_scored": len(all_labels),
        "feasible_frac": sum(all_labels) / max(len(all_labels), 1),
        "pooled_auc": pooled_auc(all_scores, all_labels),
        "mean_per_state_auc": (sum(per_state_auc) / len(per_state_auc)
                               if per_state_auc else None),
        "evidence_label": "candidate-privileged oracle-labeled diagnostic",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
