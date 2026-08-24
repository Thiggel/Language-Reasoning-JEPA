#!/usr/bin/env python3
"""Is the goal-distance / energy landscape monotone along TRUE solutions?

Motivation: oracle-goal-distance planning reaches the goal but wanders
(exact-necessary .183 at depth 1 on the full-catalogue menu, decaying to .011
at depth 8), while an exact symbolic ruler is perfect at every depth.  Two
different explanations predict that equally well:

  (i) the potential is NOT MONOTONE -- distance to the encoded goal does not
      decrease along the ground-truth solution, so descending it is not even
      trying to follow the true trajectory;
  (ii) the potential IS monotone but FLAT -- it decreases with margins that are
      small next to the spread between candidates, so the argmin is noise.

These call for opposite fixes (a monotonicity/straightening penalty vs. a
contrast/projection change), and no planning number distinguishes them.  This
script measures both directly, with no planner and no generation.

Walk the ground-truth solution.  At every prefix encode the state, and report:

  A. MONOTONICITY of the goal potential along the true trajectory
     d_t = ||s_t - g||, g = encoded solved state (same L2 the
     ``oracle_distance`` scorer uses in flat_search._score).
     descend_frac, Kendall tau vs step index, relative total drop, and the
     per-step margin measured in units of the between-candidate spread
     (``margin_over_spread``) -- that ratio is what separates (i) from (ii).

  B. LOCAL DISCRIMINATION at each true state: rank the necessary next action
     against every other feasible action, under both rulers (learned energy,
     oracle distance).  top1 / percentile rank.  This is the depth-1 argmin
     quality measured without a planner or a step cap.

  C. IMAGINED vs TRUE endpoints: the same distance for the PREDICTED next
     state, to test why predicted endpoints outrank real encoded ones
     (.920 vs .617 env at depth 4).

  D. CROSS-TIME ENERGY SCALE: E(s_t, s_t+1, s_0, 1) as a function of t.  All
     training comparisons are within a single anchor, so nothing ties these
     scales together; a systematic drift breaks any search that compares
     endpoints reached from different roots.

Every arm also gets a STEP-INDEX control: how much of each statistic is
recoverable from the step counter alone.

ORACLE DIAGNOSTIC: the goal is built from the ground-truth solved solution and
the feasible sets come from the environment.  Measurement only -- no arm here
is a deployable policy, and none of these numbers is a planning result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_flat import build_eval_dataset, load_flat_run  # noqa: E402


def kendall_tau_vs_index(vals: list[float]) -> float:
    """Kendall tau of ``vals`` against its own index.  -1 = perfectly
    decreasing, 0 = unrelated, +1 = perfectly increasing."""
    n = len(vals)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if vals[j] > vals[i]:
                conc += 1
            elif vals[j] < vals[i]:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else 0.0


def mean(xs):
    # drop None *and* NaN: a problem whose every step had a single feasible
    # action contributes no ranking sample and must not poison the aggregate
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else float("nan")


@torch.no_grad()
def encode_prefix(model, hist: list[int], device, max_len: int):
    toks = torch.tensor(hist[-max_len:], dtype=torch.long,
                        device=device).unsqueeze(0)
    return model.encode(toks)[0, -1]


def _ln(x):
    return F.layer_norm(x, x.shape[-1:])


def _dist(a, b, metric: str):
    """a [...,D] to b [D].  raw = plain L2 (what flat_search oracle_distance
    uses).  ln_l1 = LN-L1 (what objectives/geometry.goal_distances uses, and
    the equivalence class latent_pred actually trains the predictor in)."""
    if metric == "raw":
        return (a - b).norm(dim=-1)
    return (_ln(a) - _ln(b)).abs().mean(-1)


METRICS = ("raw", "ln_l1")


@torch.no_grad()
def run_problem(model, vocab, fp, device, max_len):
    """Walk the ground-truth solution; return per-problem statistics."""
    env = fp.make_env()
    hist = [t for s in fp.prompt_sentences for t in vocab.encode(s)]

    prefixes: list[list[int]] = [list(hist)]
    steps = []  # (history_before, feasible_actions, taken_action, env_clone)
    guard = 0
    while not env.solved and guard < 64:
        feas = env.feasible_actions()
        if not feas:
            break
        # the same necessary-first rule flat_search._goal_vector uses
        nxt = [q for q in feas if q in fp.necessary] or list(feas)
        q = sorted(nxt)[0]
        steps.append((list(hist), sorted(feas), q, env.clone()))
        hist = hist + vocab.encode(env.action_text(q))
        hist = hist + vocab.encode(env.step(q))
        prefixes.append(list(hist))
        guard += 1
    if not env.solved or len(steps) < 2:
        return None

    S = torch.stack([encode_prefix(model, h, device, max_len)
                     for h in prefixes], 0).float()   # [T+1, D]
    g, s0 = S[-1], S[0]
    T = len(steps)
    nec = set(fp.necessary)

    # candidate endpoints / energies at every true state, computed once
    per_step = []
    for t, (h_before, feas, taken, probe) in enumerate(steps):
        phrases = [vocab.encode(probe.action_text(a)) for a in feas]
        codes = model.encode_candidates_in_context(
            h_before, phrases, device).float()
        K = codes.shape[0]
        root = S[t].unsqueeze(0).expand(K, -1)
        ends = model.predict(root, codes).float()
        e = model.energy(root, ends, s0.unsqueeze(0).expand(K, -1),
                         1.0).float().view(-1)
        per_step.append((feas, taken, ends, e, K))

    res = {"n_steps": T, "n_necessary": len(fp.necessary)}

    # scale diagnostic: are predicted states even on the encoder's scale?
    enc_norm = float(S[:-1].norm(dim=-1).mean().item())
    pred_norm = float(torch.stack(
        [ps[2].norm(dim=-1).mean() for ps in per_step]).mean().item())
    res["encoder_state_norm"] = enc_norm
    res["predicted_state_norm"] = pred_norm
    res["pred_over_enc_norm"] = pred_norm / enc_norm if enc_norm else float("nan")

    for metric in METRICS:
        d_list = _dist(S, g, metric).tolist()
        descend = [float(d_list[t + 1] < d_list[t]) for t in range(T)]
        tru_top1, tru_pct, nec_top1 = [], [], []
        pred_descend, pred_vs_true, margins = [], [], []
        e_top1, e_pct, e_nec = [], [], []
        energy_by_t = []
        for t, (feas, taken, ends, e, K) in enumerate(per_step):
            dist = _dist(ends, g, metric)
            j = feas.index(taken)
            if K > 1:
                r_d = int((dist < dist[j]).sum().item())
                tru_top1.append(float(r_d == 0))
                tru_pct.append(r_d / (K - 1))
                nec_top1.append(float(feas[int(dist.argmin())] in nec))
                r_e = int((e < e[j]).sum().item())
                e_top1.append(float(r_e == 0))
                e_pct.append(r_e / (K - 1))
                e_nec.append(float(feas[int(e.argmin())] in nec))
                spread = float(dist.std().item())
                if spread > 0:
                    margins.append((d_list[t] - float(dist[j].item())) / spread)
            dp = float(dist[j].item())
            pred_descend.append(float(dp < d_list[t]))
            pred_vs_true.append(float(dp < d_list[t + 1]))
            energy_by_t.append(float(e[j].item()))
        m = f"_{metric}"
        res[f"descend_frac{m}"] = mean(descend)
        res[f"descend_frac_excl_last{m}"] = mean(descend[:-1])
        res[f"tau_vs_index{m}"] = kendall_tau_vs_index(d_list)
        res[f"margin_over_spread{m}"] = mean(margins)
        res[f"true_action_top1_distance{m}"] = mean(tru_top1)
        res[f"true_action_pct_distance{m}"] = mean(tru_pct)
        res[f"argmin_is_necessary_distance{m}"] = mean(nec_top1)
        res[f"predicted_descend_frac{m}"] = mean(pred_descend)
        res[f"predicted_closer_than_true_frac{m}"] = mean(pred_vs_true)
        # energy rows do not depend on the metric; stored once under raw
        if metric == "raw":
            res["true_action_top1_energy"] = mean(e_top1)
            res["true_action_pct_energy"] = mean(e_pct)
            res["argmin_is_necessary_energy"] = mean(e_nec)
            res["energy_tau_vs_index"] = kendall_tau_vs_index(energy_by_t)
            res["energy_by_t"] = energy_by_t
        res[f"d_by_t{m}"] = d_list
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--split-seed", type=int, default=2, help="2 = val")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-op", type=int, default=None)
    ap.add_argument("--max-edge", type=int, default=None)
    ap.add_argument("--op-lo", type=int, default=None)
    ap.add_argument("--op-hi", type=int, default=None)
    args = ap.parse_args()

    model, vocab, cfg = load_flat_run(args.ckpt, args.device)
    ds, caps = build_eval_dataset(
        cfg, vocab, args.n_problems, args.split_seed,
        max_op=args.max_op, max_edge=args.max_edge,
        op_lo=args.op_lo, op_hi=args.op_hi,
    )
    max_len = model.max_len if hasattr(model, "max_len") else 4096

    rows = []
    for i in range(args.n_problems):
        fp, _ = ds.problem(i)
        r = run_problem(model, vocab, fp, args.device, max_len)
        if r is not None:
            rows.append(r)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{args.n_problems}] usable={len(rows)}", flush=True)

    keys = [
        "encoder_state_norm", "predicted_state_norm", "pred_over_enc_norm",
        "true_action_top1_energy", "true_action_pct_energy",
        "argmin_is_necessary_energy", "energy_tau_vs_index", "n_steps",
    ]
    for metric in METRICS:
        m = f"_{metric}"
        keys += [f"descend_frac{m}", f"descend_frac_excl_last{m}",
                 f"tau_vs_index{m}", f"margin_over_spread{m}",
                 f"true_action_top1_distance{m}", f"true_action_pct_distance{m}",
                 f"argmin_is_necessary_distance{m}",
                 f"predicted_descend_frac{m}",
                 f"predicted_closer_than_true_frac{m}"]
    agg = {k: mean([r[k] for r in rows]) for k in keys}
    # step-index control for the monotonicity statistics: a potential that is
    # exactly the step counter scores descend_frac 1.0 / tau -1.0 by
    # construction, so these are the ceilings the learned rulers are read
    # against, not floors to beat.
    agg["step_index_ceiling_descend_frac"] = 1.0
    agg["step_index_ceiling_tau"] = -1.0
    agg["n_problems_usable"] = len(rows)

    out = {
        "protocol": {
            "ckpt": args.ckpt, "n_problems": args.n_problems,
            "split_seed": args.split_seed, "caps": caps,
            "metric": "both raw L2 (flat_search oracle_distance) and LN-L1 (objectives/geometry, the space latent_pred trains in)",
            "note": "ORACLE DIAGNOSTIC: goal and feasible sets come from the "
                    "environment.  Measurement only, not a planning result.",
        },
        "aggregate": agg,
        "per_problem": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(agg, indent=1))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
