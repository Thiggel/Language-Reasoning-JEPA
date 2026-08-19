"""Energy-head feasibility AUC on FAITHFUL iGSM (diagnostic).

Twin of ``probe_faithful_cycle_auc.py``: same oracle-driven trajectory walk,
same catalogue candidates and oracle feasible/infeasible labels, but the
score of candidate ``u`` at state ``s`` is ``-Energy(predictor(s, u))``
through the SAME head and code path the training-time ``_geo_rank`` /
``energy_cf_feasibility_rank`` contrast uses
(``DiscourseJEPA._candidate_energy``; HorizonEnergyHead at horizon 1 for
``geo_rank_score_mode=horizon``).  Successors are imagined with the true
causal prefix (``core._predict_counterfactuals``), so causal-history
predictors are scored exactly as in training; Markov predictors reduce to
``predictor(s, u)``.  Higher score should mean "feasible" if the Energy head
has learned feasibility.

Usage:
  .venv/bin/python scripts/probe_faithful_energy_auc.py --ckpt <best.pt> \
      --n-problems 24 --max-op 32 --max-edge 40 --op-lo 3 --op-hi 32 \
      --nec-lo 8 --nec-hi 15 --device cpu --out out.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from textjepa.planning.faithful_search import FaithfulPlanner
from textjepa.data.faithful import FaithfulEnv
from textjepa.utils.checkpoint import build_dataset, load_run

from probe_faithful_cycle_auc import pooled_auc  # same AUC implementation


def candidate_energies(model, s0, prev_states, action_codes, cand_codes):
    """Energies [C] of every candidate at the last state of the trajectory.

    ``prev_states`` [1, T+1, D]: s0 followed by the state after each executed
    step; ``action_codes`` [T, da]: executed action codes; ``cand_codes``
    [C, da].  Mirrors ``_geo_rank``: the anchor is the final position, the
    candidates are its alternatives, the executed prefix is teacher-forced.
    """
    T1 = prev_states.shape[1]
    C, da = cand_codes.shape
    acts = torch.cat(
        [action_codes, action_codes.new_zeros(1, da)], dim=0
    ).unsqueeze(0)  # [1, T+1, da]  (dummy at the anchor; replaced by alts)
    alts = acts.new_zeros(1, T1, C, da)
    alts[0, -1] = cand_codes
    mask = torch.ones(1, T1, dtype=torch.bool, device=prev_states.device)
    succ = model.core._predict_counterfactuals(
        prev_states, acts, alts, mask
    )[:, -1]  # [1, C, D]
    current = prev_states[:, -1]
    return model._candidate_energy(current, succ, s0).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-problems", type=int, default=24)
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
            s0 = planner._state(pt, pm, [])
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
                catalogue = [q for q in env.fp.action_order
                             if q not in env.resolved]
                if len(catalogue) < 2:
                    break
                feas = set(env.feasible_actions())
                phrases = [env.action_text(q) for q in catalogue]
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
