"""Energy-head feasibility AUC on STYLIZED iGSM (diagnostic).

Twin of ``probe_stylized_cycle_auc.py`` with the scoring rule of
``probe_faithful_energy_auc.py``: candidate ``u`` at state ``s`` scores
``-Energy(predictor(s, u))`` through ``DiscourseJEPA._candidate_energy``
(the ``_geo_rank`` path; HorizonEnergyHead at horizon 1 for
``geo_rank_score_mode=horizon``).  Labels are oracle feasibility.

Usage:
  .venv/bin/python scripts/probe_stylized_energy_auc.py --ckpt <best.pt> \
      --n-problems 24 --device cpu --out out.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import action_phrase, prompt_sentences
from textjepa.planning.search import LatentPlanner
from textjepa.utils.checkpoint import build_dataset, load_run

from probe_faithful_cycle_auc import pooled_auc
from probe_faithful_energy_auc import candidate_energies


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
            s0 = planner._s0(pt, pm)
            while not env.solved:
                if step_texts:
                    st = planner._tokens(step_texts)
                    sm = torch.ones(1, st.shape[1], dtype=torch.bool,
                                    device=device)
                    _, states = model.encode_states(pt, pm, st, sm)
                    prev_states = torch.cat([s0.unsqueeze(1), states], 1)
                    action_codes = model.encode_actions(
                        st.squeeze(0).unsqueeze(1)
                    ).squeeze(1)
                else:
                    prev_states = s0.unsqueeze(1)
                    action_codes = s0.new_zeros(0, model.core.d_action)
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
                energies = candidate_energies(
                    model, s0, prev_states, action_codes, codes
                )
                scores = (-energies).tolist()
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
        "score": "-Energy(predictor(s,u)) via DiscourseJEPA._candidate_energy",
        "geo_rank_score_mode": getattr(model, "geo_rank_score_mode", None),
        "causal_predictor": bool(
            getattr(model.core.predictor, "causal_sequence", False)
        ),
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
