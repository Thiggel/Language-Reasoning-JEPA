"""ORACLE DIAGNOSTIC (evaluation-only): goal-state multimodality measurement.

Hypothesis under test: a faithful-iGSM problem admits many valid solution
ORDERS (topological orders of the necessary steps); each order renders a
different terminal text and hence a different encoded goal point.  If the
latent ruler points at ONE reference terminal, progress along another valid
order need not reduce distance to it — which would explain why even a TRUE
re-encoded-state + oracle-distance planner fails at depth.

Per validation problem we take the dataset's reference solution order, sample
K alternative uniformly-random topological orders of the SAME necessary
actions (each verified to solve the env), render each canonically, encode all
intermediate + terminal states with the checkpoint's (student) encoder, and
measure, in both LN-L1 (the planning metric) and raw L2:

  a. goal spread            mean pairwise distance among the same problem's
                            alternative terminals (unique orders only)
  b. cross-problem baseline mean distance between reference terminals of
                            DIFFERENT problems in the same depth bucket
  c. ruler scale            mean start-state -> reference-terminal distance
  d. descent curves         distance to the REFERENCE terminal vs. step index
                            for (i) the reference trajectory, (ii) alt-order
                            trajectories, (iii) random feasible trajectories

Verdict: spread/cross and spread/scale per depth bucket.  spread ~ cross =>
same-problem alternative terminals are as far apart as unrelated states
(multimodality CONVICTED); spread << scale => terminals cluster (acquitted).

Usage:
    .venv/bin/python scripts/measure_goal_multimodality.py \
        --ckpt runs/autonomy/intent_phrase/2026-08-25-alex-results/scr-base/best.pt \
        --out-dir runs/autonomy/intent_phrase/2026-08-26-goal-multimodality-v1 \
        --device cuda:2
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_flat import load_flat_run  # noqa: E402

from textjepa.data.faithful import FaithfulDataset  # noqa: E402
from textjepa.planning.flat_search import goal_distance  # noqa: E402
from textjepa.utils import seed_everything  # noqa: E402

BUCKETS = {"2-3": (2, 3), "4-5": (4, 5), "6-8": (6, 8)}


def bucket_of(depth: int):
    for name, (lo, hi) in BUCKETS.items():
        if lo <= depth <= hi:
            return name
    return None


def reference_trace(fp, rng) -> list:
    """The dataset's own reference solution order (distractor_prob=0):
    replay FaithfulDataset.__getitem__'s trace loop, which draws
    rng.choice among currently-feasible NECESSARY actions."""
    env = fp.make_env()
    trace = []
    while not env.solved:
        nec = [q for q in env.feasible_actions() if q in fp.necessary]
        q = rng.choice(nec)
        trace.append(q)
        env.step(q)
    return trace


def random_topo_order(fp, rng: random.Random) -> list:
    """Uniform-at-each-step random topological order of the necessary set."""
    env = fp.make_env()
    order = []
    while not env.solved:
        nec = [q for q in env.feasible_actions() if q in fp.necessary]
        assert nec, "necessary set has no feasible member before solved"
        q = rng.choice(nec)
        order.append(q)
        env.step(q)
    assert env.solved and set(order) == fp.necessary
    return order


def random_feasible_traj(fp, rng: random.Random, length: int) -> list:
    """Any-feasible-action trajectory of (up to) the reference length."""
    env = fp.make_env()
    seq = []
    for _ in range(length):
        feas = env.feasible_actions()
        if not feas:
            break
        q = rng.choice(feas)
        seq.append(q)
        env.step(q)
    return seq


def execute_and_tokenize(fp, vocab, prompt_tokens: list, seq: list):
    """Execute seq in a fresh env (canonical rendering); return token
    histories after 0..len(seq) steps, plus whether the env ended solved."""
    env = fp.make_env()
    hist = list(prompt_tokens)
    hists = [list(hist)]
    for q in seq:
        hist = hist + vocab.encode(env.action_text(q))
        hist = hist + vocab.encode(env.step(q))
        hists.append(list(hist))
    return hists, env.solved


@torch.no_grad()
def encode_batch(model, vocab, device, max_len: int, histories: list[list[int]]):
    """Last-token student-encoder state per history (mirrors
    FlatPlanner._encode_batch)."""
    hs = [h[-max_len:] for h in histories]
    outs = []
    for st in range(0, len(hs), 32):
        chunk = hs[st: st + 32]
        L = max(len(h) for h in chunk)
        toks = torch.full((len(chunk), L), vocab.pad_id, dtype=torch.long,
                          device=device)
        for i, h in enumerate(chunk):
            toks[i, : len(h)] = torch.tensor(h, dtype=torch.long, device=device)
        h = model.encode(toks)
        idx = torch.tensor([len(x) - 1 for x in chunk], device=device)
        outs.append(h[torch.arange(h.shape[0], device=device), idx])
    return torch.cat(outs, 0)


def dist(a, b, metric):
    """goal_distance wrapper for two single vectors / [N,D] vs [D]."""
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return goal_distance(a, b, metric)


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k-alt", type=int, default=8)
    ap.add_argument("--k-random", type=int, default=8)
    ap.add_argument("--per-bucket", type=int, default=20)
    ap.add_argument("--split-seed", type=int, default=2, help="2 = val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan-limit", type=int, default=3000)
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, vocab, cfg = load_flat_run(args.ckpt, str(device))
    model = model.float().eval()
    max_len = int(cfg["model"]["max_len"])
    dc = cfg["data"]

    ds = FaithfulDataset(
        vocab, size=args.scan_limit, seed=args.split_seed,
        max_op=dc["max_op"], max_edge=dc["max_edge"],
        op_range=tuple(dc["op_range"]), distractor_prob=0.0,
    )

    # ---- select ~per-bucket problems per depth bucket -----------------
    chosen = defaultdict(list)  # bucket -> [(index, fp, rng, depth)]
    for i in range(args.scan_limit):
        if all(len(v) >= args.per_bucket for v in chosen.values()) \
                and len(chosen) == len(BUCKETS):
            break
        fp, rng = ds.problem(i)
        d = len(fp.necessary)
        b = bucket_of(d)
        if b is None or len(chosen[b]) >= args.per_bucket:
            continue
        chosen[b].append((i, fp, rng, d))
    n_total = sum(len(v) for v in chosen.values())
    print(f"selected {n_total} problems: "
          + ", ".join(f"{b}:{len(v)}" for b, v in sorted(chosen.items())))

    metrics = ("ln_l1", "raw")
    per_problem = []
    # bucket -> list of terminal vectors (reference), for cross-problem
    ref_terminals = defaultdict(list)

    for b in sorted(chosen):
        for idx, fp, rng, depth in chosen[b]:
            prompt_tokens = [t for s in fp.prompt_sentences
                            for t in vocab.encode(s)]
            ref = reference_trace(fp, rng)
            assert set(ref) == fp.necessary and len(ref) == depth

            alt_orders, seen = [], {tuple(ref)}
            arng = random.Random(f"gm-alt:{args.split_seed}:{idx}")
            for _ in range(args.k_alt):
                o = random_topo_order(fp, arng)
                alt_orders.append(o)
                seen.add(tuple(o))
            n_unique_orders = len(seen)

            rrng = random.Random(f"gm-rand:{args.split_seed}:{idx}")
            rand_trajs = [random_feasible_traj(fp, rrng, depth)
                         for _ in range(args.k_random)]

            # tokenize + verify
            groups = {}  # name -> list of (hists, solved)
            groups["reference"] = [execute_and_tokenize(fp, vocab,
                                                        prompt_tokens, ref)]
            groups["alt"] = [execute_and_tokenize(fp, vocab, prompt_tokens, o)
                            for o in alt_orders]
            groups["random"] = [execute_and_tokenize(fp, vocab,
                                                     prompt_tokens, t)
                               for t in rand_trajs]
            assert groups["reference"][0][1], "reference did not solve"
            assert all(s for _, s in groups["alt"]), "alt order did not solve"

            # encode everything for this problem in one batch
            flat, spans = [], {}
            for name, items in groups.items():
                spans[name] = []
                for hists, _ in items:
                    spans[name].append((len(flat), len(flat) + len(hists)))
                    flat.extend(hists)
            enc = encode_batch(model, vocab, device, max_len, flat)

            ref_states = enc[spans["reference"][0][0]:
                             spans["reference"][0][1]]
            start, ref_term = ref_states[0], ref_states[-1]
            ref_terminals[b].append(ref_term.cpu())

            # terminals of unique alternative orders (dedupe by order tuple)
            uniq_terms, seen2 = [ref_term], {tuple(ref)}
            for o, (lo, hi) in zip(alt_orders, spans["alt"]):
                if tuple(o) not in seen2:
                    seen2.add(tuple(o))
                    uniq_terms.append(enc[hi - 1])

            rec = {"index": idx, "depth": depth, "bucket": b,
                   "n_unique_orders": n_unique_orders,
                   "answer": fp.answer}
            for m in metrics:
                # a. goal spread (pairwise among unique terminals)
                pairs = [float(dist(u, v, m)) for u, v in
                         itertools.combinations(uniq_terms, 2)]
                rec[f"goal_spread_{m}"] = mean(pairs) if pairs else 0.0
                # c. ruler scale
                rec[f"start_to_goal_{m}"] = float(dist(start, ref_term, m))
                # d. descent curves (distance to REFERENCE terminal)
                for name in ("reference", "alt", "random"):
                    curves = []
                    for lo, hi in spans[name]:
                        d_vec = goal_distance(enc[lo:hi], ref_term, m)
                        curves.append([float(x) for x in d_vec])
                    rec[f"curve_{name}_{m}"] = curves
            per_problem.append(rec)
            print(f"[{b}] idx={idx} depth={depth} "
                  f"uniq={n_unique_orders} "
                  f"spread_ln={rec['goal_spread_ln_l1']:.4f} "
                  f"s2g_ln={rec['start_to_goal_ln_l1']:.4f}")

    # ---- aggregate ----------------------------------------------------
    summary = {}
    for b in sorted(chosen):
        rows = [r for r in per_problem if r["bucket"] == b]
        agg = {"n_problems": len(rows),
               "depths": sorted(r["depth"] for r in rows),
               "mean_unique_orders": mean([r["n_unique_orders"] for r in rows])}
        for m in metrics:
            # b. cross-problem baseline within the bucket
            terms = ref_terminals[b]
            cross = [float(dist(u, v, m)) for u, v in
                     itertools.combinations(terms, 2)]
            spread = mean([r[f"goal_spread_{m}"] for r in rows])
            scale = mean([r[f"start_to_goal_{m}"] for r in rows])
            agg[f"goal_spread_{m}"] = spread
            agg[f"cross_problem_{m}"] = mean(cross)
            agg[f"start_to_goal_{m}"] = scale
            agg[f"verdict_spread_over_cross_{m}"] = spread / mean(cross)
            agg[f"verdict_spread_over_scale_{m}"] = spread / scale
            # descent curves averaged by absolute step index
            for name in ("reference", "alt", "random"):
                by_idx = defaultdict(list)
                norm_by_idx = defaultdict(list)
                finals = []
                for r in rows:
                    for c in r[f"curve_{name}_{m}"]:
                        d0 = c[0] if c[0] > 0 else float("nan")
                        for t, v in enumerate(c):
                            by_idx[t].append(v)
                            norm_by_idx[t].append(v / d0)
                        finals.append(c[-1] / d0)
                agg[f"descent_{name}_{m}"] = [
                    round(mean(by_idx[t]), 5) for t in sorted(by_idx)]
                agg[f"descent_norm_{name}_{m}"] = [
                    round(mean(norm_by_idx[t]), 4) for t in sorted(norm_by_idx)]
                agg[f"final_over_start_{name}_{m}"] = mean(finals)
        summary[b] = agg

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "evidence_label": (
            "ORACLE DIAGNOSTIC (evaluation-only): uses the environment's "
            "necessary-action set and true executor to construct alternative "
            "valid solution orders; never a training signal or headline row"),
        "protocol": {
            "ckpt": args.ckpt, "split_seed": args.split_seed,
            "k_alt": args.k_alt, "k_random": args.k_random,
            "per_bucket": args.per_bucket, "encoder": "student",
            "rendering": "canonical (seq symbol naming, default)",
            "precision": "fp32",
            "caps": {"max_op": dc["max_op"], "max_edge": dc["max_edge"],
                     "op_range": list(dc["op_range"])},
            "metrics": list(metrics),
        },
        "summary": summary,
        "per_problem": per_problem,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {out_dir/'results.json'}")
    for b, agg in summary.items():
        for m in metrics:
            print(f"bucket {b} [{m}] spread={agg[f'goal_spread_{m}']:.4f} "
                  f"cross={agg[f'cross_problem_{m}']:.4f} "
                  f"scale={agg[f'start_to_goal_{m}']:.4f} "
                  f"spread/cross={agg[f'verdict_spread_over_cross_{m}']:.3f} "
                  f"spread/scale={agg[f'verdict_spread_over_scale_{m}']:.3f}")


if __name__ == "__main__":
    main()
