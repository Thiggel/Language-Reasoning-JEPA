"""ORACLE DIAGNOSTIC (evaluation-only): trained-probe progress-ranking test.

The decisive fork behind the depth cliff: the latent distance-to-goal ruler is
a LAST-MILE detector (flat for the first ~60-70% of a solution, collapsing only
in the final 2-3 steps).  Is progress information ABSENT from the frozen
encoder states, or PRESENT but unreadable by the LN-L1 ruler / energy head?

Protocol.  On faithful hard21 validation problems (seed 2, canonical
rendering, fp32 encode, scr-base checkpoint) we roll along the reference
solution; at every step t we execute + re-encode (true env, exactly like
scripts/measure_goal_multimodality.py) the TRUE next state after the necessary
action and the next states after K sampled random feasible NON-necessary
actions.  On the frozen encodings we train small probes with a pairwise
ranking loss (necessary candidate must outscore each random candidate from the
same state) on ~300 problems, and evaluate pairwise ranking accuracy on ~150
DISJOINT problems, bucketed by STEPS-TO-GO (1, 2, 3, 4-5, 6+).

Probes: linear and 2-layer MLP over [state, candidate_next, goal]; a no-goal
variant [state, candidate_next]; a state-blind control [0, candidate_next,
goal].  Baseline: untrained LN-L1 distance-to-goal ranking in the same
buckets.  Control: the whole pipeline repeated on an UNTRAINED
same-architecture encoder.

Labels come from the environment's necessary-action knowledge: this is oracle
material, evaluation-only, never a component or training signal.

Verdict logic: trained probes ~chance at steps-to-go >= 3  => information
ABSENT from the geometry (encoder objective must change).  Probes high
(>= 0.8) where LN-L1 falls toward chance => information PRESENT, readout
broken (fixable at the head level).

Usage:
    .venv/bin/python scripts/probe_progress_ranking.py \
        --ckpt runs/autonomy/intent_phrase/2026-08-25-alex-results/scr-base/best.pt \
        --out-dir runs/autonomy/intent_phrase/2026-08-26-progress-probe-v1 \
        --device cuda:2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_flat import load_flat_run  # noqa: E402

from textjepa.data.faithful import FaithfulDataset  # noqa: E402
from textjepa.models.flat_intent_jepa import FlatIntentJEPA  # noqa: E402
from textjepa.planning.flat_search import goal_distance  # noqa: E402
from textjepa.utils import seed_everything  # noqa: E402

BUCKETS = ("1", "2", "3", "4-5", "6+")


def stg_bucket(steps_to_go: int) -> str:
    if steps_to_go <= 3:
        return str(steps_to_go)
    if steps_to_go <= 5:
        return "4-5"
    return "6+"


def reference_trace(fp, rng) -> list:
    """Replay the dataset's own reference solution order (distractor_prob=0)."""
    env = fp.make_env()
    trace = []
    while not env.solved:
        nec = [q for q in env.feasible_actions() if q in fp.necessary]
        q = rng.choice(nec)
        trace.append(q)
        env.step(q)
    return trace


@torch.no_grad()
def encode_batch(model, vocab, device, max_len: int, histories: list[list[int]],
                 chunk: int = 16):
    """Last-token student-encoder state per history (fp32, mirrors
    FlatPlanner._encode_batch / measure_goal_multimodality)."""
    hs = [h[-max_len:] for h in histories]
    outs = []
    for st in range(0, len(hs), chunk):
        ch = hs[st: st + chunk]
        L = max(len(h) for h in ch)
        toks = torch.full((len(ch), L), vocab.pad_id, dtype=torch.long,
                          device=device)
        for i, h in enumerate(ch):
            toks[i, : len(h)] = torch.tensor(h, dtype=torch.long, device=device)
        h = model.encode(toks)
        idx = torch.tensor([len(x) - 1 for x in ch], device=device)
        outs.append(h[torch.arange(h.shape[0], device=device), idx].float().cpu())
    return torch.cat(outs, 0)


def collect_problem(fp, rng, vocab, k_neg: int, neg_rng: random.Random):
    """Walk the reference solution; per step return token histories for the
    state, the true next state (necessary action), and next states after
    k_neg random feasible non-necessary actions.

    Returns (histories, records): histories is a flat list of token lists;
    records is a list of dicts with indices into histories:
      {"steps_to_go": s, "state": i, "pos": j, "negs": [..]}
    plus the goal history index under key "goal_index" in the last record slot.
    """
    ref = reference_trace(fp, rng)
    depth = len(ref)
    prompt = [t for s in fp.prompt_sentences for t in vocab.encode(s)]

    env = fp.make_env()
    hist = list(prompt)
    histories = [list(hist)]          # index 0 = start state
    state_idx = 0
    records = []
    pending = []  # (steps_to_go, state_idx, neg_indices_placeholder)

    for t, q in enumerate(ref):
        # negatives from this state: feasible, NOT in the necessary set
        feas = env.feasible_actions()
        neg_pool = [a for a in feas if a not in fp.necessary]
        negs = neg_rng.sample(neg_pool, min(k_neg, len(neg_pool)))
        neg_idx = []
        for a in negs:
            c = env.clone()
            nh = hist + vocab.encode(c.action_text(a))
            nh = nh + vocab.encode(c.step(a))
            histories.append(nh)
            neg_idx.append(len(histories) - 1)
        # true next state (necessary action) on the main env
        hist = hist + vocab.encode(env.action_text(q))
        hist = hist + vocab.encode(env.step(q))
        histories.append(list(hist))
        pos_idx = len(histories) - 1
        if neg_idx:
            records.append({"steps_to_go": depth - t, "state": state_idx,
                            "pos": pos_idx, "negs": neg_idx})
        state_idx = pos_idx
    assert env.solved
    goal_idx = state_idx  # reference terminal
    return histories, records, goal_idx, depth


class MLPProbe(torch.nn.Module):
    def __init__(self, d_in: int, hidden: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_features(pairs, feats, variant: str):
    """pairs: list of (state_i, cand_i, goal_i, label-irrelevant); returns
    the concatenated feature tensor for the given probe variant."""
    s = feats[[p[0] for p in pairs]]
    c = feats[[p[1] for p in pairs]]
    g = feats[[p[2] for p in pairs]]
    if variant == "full":
        return torch.cat([s, c, g], -1)
    if variant == "no_goal":
        return torch.cat([s, c], -1)
    if variant == "state_blind":
        return torch.cat([torch.zeros_like(s), c, g], -1)
    raise ValueError(variant)


def train_probe(kind, x_pos, x_neg, seed=0, epochs=400, lr=1e-3, wd=1e-4):
    """Pairwise logistic ranking loss: pos must outscore its matched neg.
    x_pos/x_neg: [N, D] matched rows (one row per pos-neg pair)."""
    torch.manual_seed(seed)
    d = x_pos.shape[1]
    probe = (torch.nn.Linear(d, 1) if kind == "linear"
             else MLPProbe(d)).to(x_pos.device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)
    n = x_pos.shape[0]
    bs = min(4096, n)
    for ep in range(epochs):
        perm = torch.randperm(n, device=x_pos.device)
        for st in range(0, n, bs):
            idx = perm[st: st + bs]
            sp = probe(x_pos[idx]).squeeze(-1)
            sn = probe(x_neg[idx]).squeeze(-1)
            loss = torch.nn.functional.softplus(sn - sp).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe.eval()


@torch.no_grad()
def eval_probe(probe, x_pos, x_neg, buckets):
    """Per-bucket pairwise accuracy (pos outscores neg; chance 0.5)."""
    sp = probe(x_pos).squeeze(-1)
    sn = probe(x_neg).squeeze(-1)
    correct = (sp > sn).float().cpu()
    out = {}
    for b in BUCKETS:
        m = [i for i, bb in enumerate(buckets) if bb == b]
        out[b] = {"n_pairs": len(m),
                  "acc": round(float(correct[m].mean()), 4) if m else None}
    out["overall"] = {"n_pairs": len(buckets),
                      "acc": round(float(correct.mean()), 4)}
    return out


def build_pairs(records_by_problem, feat_offset):
    """Flatten per-problem records into matched (pos, neg) index pairs with
    global feature indices; returns list of (state, pos, goal, neg, bucket)."""
    pairs = []
    for off, records, goal in records_by_problem:
        for r in records:
            b = stg_bucket(r["steps_to_go"])
            for ni in r["negs"]:
                pairs.append((off + r["state"], off + r["pos"], off + goal,
                              off + ni, b))
    return pairs


def encode_split(model, vocab, device, max_len, ds, indices, k_neg, tag):
    """Encode all histories for a list of problem indices (re-fetched from
    the dataset so the per-problem rng state is identical across passes);
    returns (features tensor [N,D] on cpu, records_by_problem)."""
    all_feats, rbp = [], []
    n_hist = 0
    for j, idx in enumerate(indices):
        fp, rng = ds.problem(idx)
        neg_rng = random.Random(f"probe-neg:{idx}")
        hists, records, goal_idx, depth = collect_problem(
            fp, rng, vocab, k_neg, neg_rng)
        feats = encode_batch(model, vocab, device, max_len, hists)
        rbp.append((n_hist, records, goal_idx))
        n_hist += len(hists)
        all_feats.append(feats)
        if (j + 1) % 25 == 0:
            print(f"  [{tag}] encoded {j + 1}/{len(indices)} problems "
                  f"({n_hist} states)", flush=True)
    return torch.cat(all_feats, 0), rbp


def ln_l1_baseline(feats, pairs):
    """Untrained LN-L1 distance-to-goal ranking accuracy per bucket."""
    out_correct, buckets = [], []
    for s, p, g, n, b in pairs:
        dp = float(goal_distance(feats[p].unsqueeze(0), feats[g], "ln_l1"))
        dn = float(goal_distance(feats[n].unsqueeze(0), feats[g], "ln_l1"))
        out_correct.append(1.0 if dp < dn else 0.0)
        buckets.append(b)
    res = {}
    for b in BUCKETS:
        m = [c for c, bb in zip(out_correct, buckets) if bb == b]
        res[b] = {"n_pairs": len(m),
                  "acc": round(sum(m) / len(m), 4) if m else None}
    res["overall"] = {"n_pairs": len(buckets),
                      "acc": round(sum(out_correct) / len(out_correct), 4)}
    return res


def run_probe_suite(train_feats, train_pairs, eval_feats, eval_pairs, device):
    """Train + evaluate every probe variant on one encoder's features."""
    results = {}
    # standardize features with train stats (per-dim, on raw encoder states)
    mu = train_feats.mean(0, keepdim=True)
    sd = train_feats.std(0, keepdim=True).clamp_min(1e-6)
    tf = ((train_feats - mu) / sd).to(device)
    ef = ((eval_feats - mu) / sd).to(device)

    eval_buckets = [p[4] for p in eval_pairs]
    variants = ("full", "no_goal", "state_blind")
    kinds = ("linear", "mlp")
    for variant in variants:
        xtr_pos = make_features([(p[0], p[1], p[2]) for p in train_pairs], tf, variant)
        xtr_neg = make_features([(p[0], p[3], p[2]) for p in train_pairs], tf, variant)
        xev_pos = make_features([(p[0], p[1], p[2]) for p in eval_pairs], ef, variant)
        xev_neg = make_features([(p[0], p[3], p[2]) for p in eval_pairs], ef, variant)
        for kind in kinds:
            if variant == "state_blind" and kind == "linear":
                continue  # control: MLP only
            probe = train_probe(kind, xtr_pos, xtr_neg)
            results[f"{kind}_{variant}"] = eval_probe(
                probe, xev_pos, xev_neg, eval_buckets)
            with torch.no_grad():
                tr_acc = ((probe(xtr_pos).squeeze(-1)
                           > probe(xtr_neg).squeeze(-1)).float().mean())
            results[f"{kind}_{variant}"]["train_acc"] = round(float(tr_acc), 4)
            print(f"    probe {kind}/{variant}: "
                  f"overall acc {results[f'{kind}_{variant}']['overall']['acc']}",
                  flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--k-neg", type=int, default=3)
    ap.add_argument("--depth-min", type=int, default=3)
    ap.add_argument("--depth-max", type=int, default=8)
    ap.add_argument("--split-seed", type=int, default=2, help="2 = val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan-limit", type=int, default=8000)
    ap.add_argument("--skip-untrained", action="store_true")
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

    # ---- select disjoint train/eval problems, stratified over depth -------
    depths = list(range(args.depth_min, args.depth_max + 1))
    q_train = {d: args.n_train // len(depths) for d in depths}
    q_eval = {d: args.n_eval // len(depths) for d in depths}
    train_probs, eval_probs = [], []
    tr_left = dict(q_train)
    ev_left = dict(q_eval)
    for i in range(args.scan_limit):
        if not any(tr_left.values()) and not any(ev_left.values()):
            break
        fp, rng = ds.problem(i)
        d = len(fp.necessary)
        if d not in tr_left:
            continue
        if tr_left[d] > 0:
            train_probs.append(i)
            tr_left[d] -= 1
        elif ev_left[d] > 0:
            eval_probs.append(i)
            ev_left[d] -= 1
    # top up any shortfall with remaining in-range problems (rare deep ones)
    print(f"selected train={len(train_probs)} eval={len(eval_probs)} "
          f"(shortfall train={sum(tr_left.values())} eval={sum(ev_left.values())})",
          flush=True)
    assert not (set(train_probs) & set(eval_probs)), "train/eval overlap"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "evidence_label": (
            "ORACLE DIAGNOSTIC (evaluation-only): probe labels come from the "
            "environment's necessary-action set and true executor; trained "
            "probes are diagnostic readouts of frozen encoder states, never "
            "a planner component, training signal, or headline row"),
        "protocol": {
            "ckpt": args.ckpt, "split_seed": args.split_seed,
            "n_train_problems": len(train_probs),
            "n_eval_problems": len(eval_probs),
            "k_neg": args.k_neg,
            "depth_range": [args.depth_min, args.depth_max],
            "encoder": "student", "precision": "fp32",
            "rendering": "canonical (seq symbol naming, default)",
            "caps": {"max_op": dc["max_op"], "max_edge": dc["max_edge"],
                     "op_range": list(dc["op_range"])},
            "negatives": "random feasible NON-necessary actions, executed in "
                         "a cloned true env and re-encoded",
            "probes": "pairwise logistic ranking loss on frozen features; "
                      "linear + 2-layer MLP (hidden 512); features "
                      "standardized with probe-train stats",
            "buckets": "steps-to-go at the decision state",
        },
    }

    # ---- trained encoder --------------------------------------------------
    print("encoding TRAINED encoder states...", flush=True)
    tr_feats, tr_rbp = encode_split(model, vocab, device, max_len,
                                    ds, train_probs, args.k_neg, "train")
    ev_feats, ev_rbp = encode_split(model, vocab, device, max_len,
                                    ds, eval_probs, args.k_neg, "eval")
    tr_pairs = build_pairs(tr_rbp, 0)
    ev_pairs = build_pairs(ev_rbp, 0)
    print(f"pairs: train={len(tr_pairs)} eval={len(ev_pairs)}", flush=True)

    print("LN-L1 oracle-distance baseline (no training)...", flush=True)
    payload["trained_encoder"] = {
        "ln_l1_distance_baseline": ln_l1_baseline(ev_feats, ev_pairs)}
    print("training probes on trained-encoder features...", flush=True)
    payload["trained_encoder"]["probes"] = run_probe_suite(
        tr_feats, tr_pairs, ev_feats, ev_pairs, device)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    # ---- untrained-encoder control ---------------------------------------
    if not args.skip_untrained:
        print("building UNTRAINED same-architecture encoder...", flush=True)
        torch.manual_seed(1234)
        mcfg = dict(cfg["model"])
        mcfg["init_from_lm"] = None
        mcfg["lm_detach_state"] = bool(
            dict(cfg.get("objective", {})).get("intent_prior_lm", {})
            .get("detach_state", False))
        rmodel = FlatIntentJEPA(vocab_size=len(vocab), pad_id=vocab.pad_id,
                                **mcfg).to(device).float().eval()
        utr_feats, _ = encode_split(rmodel, vocab, device, max_len,
                                    ds, train_probs, args.k_neg, "utrain")
        uev_feats, _ = encode_split(rmodel, vocab, device, max_len,
                                    ds, eval_probs, args.k_neg, "ueval")
        # same pair structure (identical histories / negatives by seed)
        payload["untrained_encoder"] = {
            "ln_l1_distance_baseline": ln_l1_baseline(uev_feats, ev_pairs)}
        print("training probes on untrained-encoder features...", flush=True)
        payload["untrained_encoder"]["probes"] = run_probe_suite(
            utr_feats, tr_pairs, uev_feats, ev_pairs, device)

    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {out_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
