"""Does the energy head separate PROGRESS from NON-PROGRESS, not just legality?

The existing ``probe_flat_energy_auc.py`` asks whether lower energy means an
action is LEGAL at the imagined state.  Legality is not goodness: a legal
action can still be useless (it resolves a variable the query does not need).
Deep search can only work if, AMONG LEGAL ACTIONS, lower energy also means the
action actually advances the solution.

Protocol (mirrors the legality probe so the two curves are comparable):
walk a random FEASIBLE trajectory; at each visited state, imagine ``d`` random
feasible actions forward (latent predictor chain) while stepping a CLONE of the
environment through the same actions; then score every catalogue action from
the imagined prefix and report, at each depth d:

* ``legality_auc``  AUC of -E over all not-yet-resolved catalogue actions,
                    labelled by legality in the cloned env   (the old curve);
* ``progress_auc``  AUC of -E restricted to the LEGAL actions only, labelled
                    by whether the action is NECESSARY for the query.

EVIDENCE LABEL: candidate-privileged, ORACLE-LABELLED DIAGNOSTIC.  The symbolic
state (feasibility, the necessary set) is used ONLY to label the measurement --
it is never a model input and never a training signal (CLAUDE.md /
projects/token_igsm/NORMATIVE_CONTRACT.md: symbolic iGSM state is
evaluation-only).
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
from plan_flat import build_eval_dataset, load_flat_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-problems", type=int, default=64)
    ap.add_argument("--depths", default="0,1,2,3,4,6,8")
    ap.add_argument("--max-op", type=int, default=None)
    ap.add_argument("--max-edge", type=int, default=None)
    ap.add_argument("--op-lo", type=int, default=None)
    ap.add_argument("--op-hi", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=2)
    ap.add_argument("--reencode", action="store_true", help=(
        "replace the PREDICTOR-IMAGINED prefix with a TRUE RE-ENCODING of the "
        "executed text after the same d actions.  Comparing the two curves "
        "decomposes the depth collapse into imagination error (re-encode "
        "curve stays up) vs head error (re-encode curve collapses too)."))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]
    device = torch.device(args.device)
    model, vocab, cfg = load_flat_run(args.ckpt, str(device))
    dataset, caps = build_eval_dataset(
        cfg, vocab, args.n_problems, args.split_seed, args.max_op,
        args.max_edge, args.op_lo, args.op_hi,
    )
    rng = random.Random(args.seed)
    leg_s = {d: [] for d in depths}
    leg_l = {d: [] for d in depths}
    pro_s = {d: [] for d in depths}
    pro_l = {d: [] for d in depths}
    per_state = {d: [] for d in depths}
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
           else torch.autocast("cpu", enabled=False))
    with torch.no_grad(), ctx:
        for i in range(args.n_problems):
            fp, _ = dataset.problem(i)
            env = fp.make_env()
            necessary = set(fp.necessary)
            history = [t for s in fp.prompt_sentences for t in vocab.encode(s)]
            h = model.encode(torch.tensor(history, device=device).unsqueeze(0))
            s0 = h[0, -1]
            while not env.solved:
                h = model.encode(torch.tensor(history, device=device).unsqueeze(0))
                state = h[0, -1]
                catalogue = [q for q in fp.action_order if q not in env.resolved]
                if len(catalogue) < 2:
                    break
                phrases = [vocab.encode(env.action_text(q)) for q in catalogue]
                codes = model.encode_candidates_in_context(history, phrases, device)
                for d in depths:
                    # imagined prefix after d random feasible actions, with a
                    # cloned env walked through the SAME actions (labels only)
                    probe = env.clone()
                    prefix = state.unsqueeze(0)
                    text2 = list(history)
                    used: list = []
                    ok = True
                    for _ in range(d):
                        f = sorted(probe.feasible_actions())
                        if not f or probe.solved:
                            ok = False
                            break
                        a = rng.choice(f)
                        j = catalogue.index(a) if a in catalogue else None
                        if j is None:
                            ok = False
                            break
                        text2 += vocab.encode(probe.action_text(a))
                        prefix = model.predict(prefix, codes[j].unsqueeze(0))
                        text2 += vocab.encode(probe.step(a))
                        used.append(a)
                    if ok and args.reencode and d > 0:
                        # ground truth for the imagined prefix: encode the
                        # actually-executed text after the same d actions
                        h2 = model.encode(
                            torch.tensor(text2, device=device).unsqueeze(0))
                        prefix = h2[0, -1].unsqueeze(0)
                    if not ok or probe.solved:
                        continue
                    cat = [q for q in catalogue if q not in used and q not in probe.resolved]
                    if len(cat) < 2:
                        continue
                    idx = torch.tensor([catalogue.index(q) for q in cat], device=device)
                    n = len(cat)
                    succ = model.predict(prefix.expand(n, -1), codes[idx])
                    e = model.energy(state.unsqueeze(0).expand(n, -1), succ,
                                     s0.unsqueeze(0).expand(n, -1), float(d + 1)).float()
                    sc = (-e).tolist()
                    feas = set(probe.feasible_actions())
                    leg_s[d] += sc
                    leg_l[d] += [int(q in feas) for q in cat]
                    # progress: among LEGAL actions only, is it necessary?
                    sub = [(s_, int(q in necessary)) for s_, q in zip(sc, cat) if q in feas]
                    labs = [b for _, b in sub]
                    if sub and 0 < sum(labs) < len(labs):
                        pro_s[d] += [s_ for s_, _ in sub]
                        pro_l[d] += labs
                        a_ = pooled_auc([s_ for s_, _ in sub], labs)
                        if a_ == a_:
                            per_state[d].append(a_)
                q = rng.choice(sorted(env.feasible_actions()))
                history += vocab.encode(env.action_text(q))
                history += vocab.encode(env.step(q))
    report = {
        "ckpt": args.ckpt, "caps": caps, "n_problems": args.n_problems,
        "reencode": bool(args.reencode),
        "evidence_label": (
            "CANDIDATE-PRIVILEGED, ORACLE-LABELLED DIAGNOSTIC: the symbolic "
            "iGSM state (feasibility + the necessary set) labels the "
            "measurement only and is never a model input."),
        "by_depth": {},
    }
    for d in depths:
        row = {
            "legality_auc": pooled_auc(leg_s[d], leg_l[d]) if leg_l[d] else None,
            "legality_n": len(leg_l[d]),
            "legal_frac": (sum(leg_l[d]) / len(leg_l[d])) if leg_l[d] else None,
            "progress_auc": pooled_auc(pro_s[d], pro_l[d]) if pro_l[d] else None,
            "progress_mean_per_state_auc": (
                sum(per_state[d]) / len(per_state[d]) if per_state[d] else None),
            "progress_n": len(pro_l[d]),
            "necessary_frac_among_legal": (
                sum(pro_l[d]) / len(pro_l[d])) if pro_l[d] else None,
        }
        report["by_depth"][str(d)] = row
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["by_depth"], indent=2))


if __name__ == "__main__":
    main()
