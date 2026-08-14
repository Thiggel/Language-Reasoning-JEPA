"""LDAD cycle-consistency feasibility AUC on STYLIZED iGSM (control probe).

Twin of ``scripts/probe_faithful_cycle_auc.py``, sharing the exact same LDAD
cycle path (``textjepa.planning.ldad_decode.delta_logits`` +
``phrase_log_probs``).  Purpose: probe-validity control for the faithful
chance result (AUC .475) — if the stylized LDAD checkpoint scores ~.9 through
THIS code path, the faithful chance result is a property of the faithful
checkpoint/data, not of the probe.

Protocol (matches the faithful probe): sample validation problems, walk a
random FEASIBLE trajectory (oracle-driven — candidate-privileged diagnostic,
not a planning result), and at every visited state score every not-yet-
resolved catalogue action with the cycle score; labels from the symbolic
feasibility oracle.  Report pooled and per-state-mean ROC AUC.

Usage:
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/probe_stylized_cycle_auc.py \
      --ckpt <stylized best.pt> --n-problems 24 --device cpu --out out.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import action_phrase, prompt_sentences
from textjepa.planning.ldad_decode import delta_logits, phrase_log_probs
from textjepa.planning.search import LatentPlanner
from textjepa.utils.checkpoint import build_dataset, load_run

from probe_faithful_cycle_auc import pooled_auc  # same AUC implementation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-problems", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, vocab, cfg = load_run(args.ckpt, device=str(device))
    dataset = build_dataset(cfg, vocab, args.split, size=args.n_problems)

    planner = LatentPlanner(model, vocab, device)
    rng = random.Random(args.seed)
    all_scores: list[float] = []
    all_labels: list[int] = []
    per_state_auc: list[float] = []

    with torch.no_grad():
        for i in range(args.n_problems):
            problem, _ = dataset.problem(i)
            env = SymbolicEnv(problem)
            prompt = prompt_sentences(problem, random.Random(args.seed + i))
            pt = planner._tokens(prompt)
            pm = torch.ones(1, pt.shape[1], dtype=torch.bool, device=device)
            step_texts: list[str] = []
            while not env.solved:
                state = planner._current_state(pt, pm, step_texts)
                resolved = env.resolved_set
                catalogue = [
                    v.idx for v in problem.vars if v.idx not in resolved
                ]
                if len(catalogue) < 2:
                    break
                feas = set(env.feasible_actions())
                phrases = [action_phrase(problem, q) for q in catalogue]
                codes = model.encode_actions(
                    planner._tokens(phrases).squeeze(0).unsqueeze(1)
                ).squeeze(1)
                scores = phrase_log_probs(
                    delta_logits(model, state, codes,
                                 context="stylized cycle AUC probe"),
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
