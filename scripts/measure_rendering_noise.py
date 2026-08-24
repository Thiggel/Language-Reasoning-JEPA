#!/usr/bin/env python3
"""A2 + drift-curve diagnostic.

Walks ground-truth solutions and measures two things the planning numbers
cannot separate:

  1. RENDERING NOISE (A2): the environment's step text draws temporary
     variable names from the global RNG, so the same action in the same state
     renders differently.  We re-render each step K times, encode each
     variant, and report the LN-L1 spread — the irreducible noise floor of
     the predictor's targets.  Compared against the mean per-step state
     movement (signal scale) and the predictor's 1-step error.

  2. DRIFT CURVE (A1): from every anchor s_t, roll the predictor k=1..8
     steps along the TRUE observed actions and measure LN-L1 to the encoded
     true s_{t+k}.  Exposure bias predicts error growing with k; rollpred-
     style supervision predicts it flattening up to the supervised k.

ORACLE DIAGNOSTIC: environment-derived, measurement only, never a planning
result.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_flat import build_eval_dataset, load_flat_run  # noqa: E402
from measure_energy_monotonicity import encode_prefix, mean  # noqa: E402


def lnl1(a, b):
    return float((F.layer_norm(a.float(), a.shape[-1:])
                  - F.layer_norm(b.float(), b.shape[-1:])).abs().mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-problems", type=int, default=50)
    ap.add_argument("--renders", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--split-seed", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    model, vocab, cfg = load_flat_run(args.ckpt, args.device)
    ds, caps = build_eval_dataset(cfg, vocab, args.n_problems,
                                  args.split_seed)
    max_len = model.max_len if hasattr(model, "max_len") else 4096
    dev = args.device

    spreads, movement, pred_err = [], [], []
    drift = {k: [] for k in range(1, args.kmax + 1)}

    for i in range(args.n_problems):
        fp, _ = ds.problem(i)
        env = fp.make_env()
        hist = [t for s in fp.prompt_sentences for t in vocab.encode(s)]
        prefixes = [list(hist)]
        acts, steps = [], []
        rng = random.Random(f"noise:{i}")
        guard = 0
        while not env.solved and guard < 64:
            feas = env.feasible_actions()
            if not feas:
                break
            nxt = [q for q in feas if q in fp.necessary] or list(feas)
            q = sorted(nxt)[0]
            # ---- A2: re-render this exact step K times ------------------
            variants = []
            for _ in range(args.renders):
                py, nps = random.getstate(), np.random.get_state()
                random.seed(rng.random())
                np.random.seed(rng.randrange(2 ** 31))
                try:
                    variants.append(vocab.encode(env.clone().step(q)))
                finally:
                    random.setstate(py)
                    np.random.set_state(nps)
            base = hist + vocab.encode(env.action_text(q))
            with torch.no_grad():
                var_states = torch.stack([
                    encode_prefix(model, base + v, dev, max_len)
                    for v in variants
                ]).float()
            pw = [lnl1(var_states[a], var_states[b])
                  for a in range(len(variants))
                  for b in range(a + 1, len(variants))]
            spreads.append(sum(pw) / len(pw))
            # ---- canonical continuation ---------------------------------
            steps.append((list(hist), q, env.clone()))
            hist = base + vocab.encode(env.step(q))
            prefixes.append(list(hist))
            acts.append(q)
            guard += 1
        if not env.solved or len(acts) < 2:
            continue
        with torch.no_grad():
            S = torch.stack([encode_prefix(model, h, dev, max_len)
                             for h in prefixes]).float()  # [T+1, D]
        T = len(acts)
        for t in range(T):
            movement.append(lnl1(S[t], S[t + 1]))
        # ---- drift curve: rollout along true actions --------------------
        with torch.no_grad():
            for t, (h_before, q, probe) in enumerate(steps):
                kcap = min(args.kmax, T - t)
                phrases = [vocab.encode(probe.action_text(a))
                           for a in acts[t:t + kcap]]
                codes = model.encode_candidates_in_context(
                    h_before, phrases, dev).float()
                state = S[t].unsqueeze(0)
                for k in range(1, kcap + 1):
                    state = model.predict(state, codes[k - 1:k])
                    err = lnl1(state[0], S[t + k])
                    drift[k].append(err)
                    if k == 1:
                        pred_err.append(err)

    agg = {
        "render_spread_lnl1": mean(spreads),
        "step_movement_lnl1": mean(movement),
        "pred_err_1step_lnl1": mean(pred_err),
        "spread_over_movement": mean(spreads) / mean(movement),
        "prederr_over_spread": mean(pred_err) / mean(spreads),
        "drift_by_k": {k: mean(v) for k, v in drift.items() if v},
        "drift_n_by_k": {k: len(v) for k, v in drift.items()},
    }
    out = {
        "protocol": {"ckpt": args.ckpt, "n_problems": args.n_problems,
                     "renders": args.renders, "caps": caps,
                     "note": "ORACLE DIAGNOSTIC: measurement only."},
        "aggregate": agg,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
