"""Feasibility AUC probes for flat-backbone intent JEPA checkpoints.

Same protocol and evidence label as ``probe_faithful_cycle_auc.py`` /
``probe_faithful_energy_auc.py``: walk a random FEASIBLE trajectory
(oracle-driven), and at every visited state score every not-yet-resolved
catalogue intent; labels come from ``env.feasible_actions()``.  Three scores
from the same checkpoint:

* ``energy_depth0``  -E(s, predictor(s, u), s_0)   (root legality);
* ``energy_depth1``  take one random feasible action a1 (oracle walk), form
                     the IMAGINED prefix p1 = predictor(s, a1) and score
                     catalogue intents u by -E(s, predictor(p1, u), s_0);
                     labels are feasibility in the env AFTER a1 -- this is
                     exactly how oracle-free lookahead scores deeper slots;
* ``cycle``          LDAD cycle score (mean token log-prob of u's own phrase
                     decoded from predictor(s, u) - s).

Usage:
  .venv/bin/python scripts/probe_flat_energy_auc.py --ckpt <pt> --n-problems 24 \
      --max-op 15 --max-edge 20 --op-lo 3 --op-hi 15 --device cuda:0 --out out.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_faithful_cycle_auc import pooled_auc  # noqa: E402
from plan_flat import load_flat_run  # noqa: E402

from textjepa.data.faithful import FaithfulDataset, FaithfulEnv  # noqa: E402
from textjepa.planning.ldad_decode import phrase_log_probs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-problems", type=int, default=24)
    ap.add_argument("--max-op", type=int, default=15)
    ap.add_argument("--max-edge", type=int, default=20)
    ap.add_argument("--op-lo", type=int, default=3)
    ap.add_argument("--op-hi", type=int, default=15)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    model, vocab, cfg = load_flat_run(args.ckpt, str(device))
    dataset = FaithfulDataset(
        vocab, size=args.n_problems, seed=args.split_seed, max_op=args.max_op,
        max_edge=args.max_edge, op_range=(args.op_lo, args.op_hi), distractor_prob=0.0,
    )
    rng = random.Random(args.seed)
    scores = {"energy_depth0": [], "energy_depth1": [], "cycle": []}
    labels = {"energy_depth0": [], "energy_depth1": [], "cycle": []}
    per_state = {k: [] for k in scores}
    has_ldad = model.observed_action_decoder is not None
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
           else torch.autocast("cpu", enabled=False))
    with torch.no_grad(), ctx:
        for i in range(args.n_problems):
            fp, _ = dataset.problem(i)
            env = FaithfulEnv(fp)
            history = [t for s in fp.prompt_sentences for t in vocab.encode(s)]
            h = model.encode(torch.tensor(history, device=device).unsqueeze(0))
            s0 = h[0, -1]
            while not env.solved:
                h = model.encode(torch.tensor(history, device=device).unsqueeze(0))
                state = h[0, -1]
                catalogue = [q for q in fp.action_order if q not in env.resolved]
                if len(catalogue) < 2:
                    break
                feas = set(env.feasible_actions())
                phrases = [vocab.encode(env.action_text(q)) for q in catalogue]
                codes = model.encode_candidates_in_context(history, phrases, device)
                n = len(catalogue)
                root = state.unsqueeze(0).expand(n, -1)
                init = s0.unsqueeze(0).expand(n, -1)
                succ = model.predict(root, codes)
                e0 = model.energy(root, succ, init, 1).float()
                lab = [int(q in feas) for q in catalogue]
                sc = (-e0).tolist()
                scores["energy_depth0"] += sc
                labels["energy_depth0"] += lab
                a = pooled_auc(sc, lab)
                if a == a:
                    per_state["energy_depth0"].append(a)
                if has_ldad:
                    logits = model.observed_action_decoder(succ - root)
                    cyc = phrase_log_probs(logits.float(), phrases).tolist()
                    scores["cycle"] += cyc
                    labels["cycle"] += lab
                    a = pooled_auc(cyc, lab)
                    if a == a:
                        per_state["cycle"].append(a)
                # depth-1: imagined prefix after a random feasible a1
                a1 = rng.choice(sorted(feas))
                i1 = catalogue.index(a1)
                p1 = succ[i1].unsqueeze(0)
                env1 = env.clone()
                env1.step(a1)
                feas1 = set(env1.feasible_actions())
                cat1 = [q for q in catalogue if q != a1]
                if len(cat1) >= 2:
                    idx1 = torch.tensor([catalogue.index(q) for q in cat1], device=device)
                    codes1 = codes[idx1]
                    n1 = len(cat1)
                    succ1 = model.predict(p1.expand(n1, -1), codes1)
                    e1 = model.energy(state.unsqueeze(0).expand(n1, -1), succ1,
                                      s0.unsqueeze(0).expand(n1, -1), 2).float()
                    lab1 = [int(q in feas1) for q in cat1]
                    sc1 = (-e1).tolist()
                    scores["energy_depth1"] += sc1
                    labels["energy_depth1"] += lab1
                    a = pooled_auc(sc1, lab1)
                    if a == a:
                        per_state["energy_depth1"].append(a)
                # oracle walk
                q = rng.choice(sorted(feas))
                history += vocab.encode(env.action_text(q))
                history += vocab.encode(env.step(q))
    report = {
        "ckpt": args.ckpt,
        "caps": {"max_op": args.max_op, "max_edge": args.max_edge, "op_range": [args.op_lo, args.op_hi]},
        "n_problems": args.n_problems,
        "evidence_label": "candidate-privileged oracle-labeled diagnostic",
    }
    for k in scores:
        if labels[k]:
            report[f"{k}_pooled_auc"] = pooled_auc(scores[k], labels[k])
            report[f"{k}_mean_per_state_auc"] = (
                sum(per_state[k]) / len(per_state[k]) if per_state[k] else None)
            report[f"{k}_n_scored"] = len(labels[k])
            report[f"{k}_feasible_frac"] = sum(labels[k]) / len(labels[k])
    # backwards-compatible aliases for the watcher/curve readers
    report["pooled_auc"] = report.get("energy_depth0_pooled_auc")
    report["cycle_pooled_auc"] = report.get("cycle_pooled_auc")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
