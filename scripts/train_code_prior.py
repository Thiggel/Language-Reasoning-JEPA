"""Fit a state-conditioned DISCRETE prior over action codes on a FROZEN
flat-intent-JEPA checkpoint.

    .venv/bin/python scripts/train_code_prior.py \
        --ckpt <best.pt> --n-codes 256 --out prior.pt

Two learned objects, both self-supervised on the actions that ACTUALLY
OCCURRED in the demonstrated traces (the same signal the LM baselines train
on): a codebook over the checkpoint's own action vectors, and p(code | state).
The target code index is a LEARNED latent, never an environment label -- no
symbolic state, no steps-to-go, no feasibility supervision (CLAUDE.md).

Nothing in the JEPA is retrained and no config key is added, so every existing
run and config behaves exactly as before.  Reuses the (state, action, history)
cache written by ``scripts/train_action_decoder.py`` when given ``--cache``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn.functional import cross_entropy as nn_ce

from textjepa.planning.code_prior import CodePrior, CodePriorConfig
from textjepa.planning.flat_search import FlatPlanner
from textjepa.utils import seed_everything

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_flat import build_eval_dataset, load_flat_run  # noqa: E402
from train_action_decoder import (cache_pairs, context_hiddens,  # noqa: E402
                                  pad_batch)


@torch.no_grad()
def states_for(planner, hist, batch=32):
    """Recompute the frozen state at each cached history."""
    out = []
    for s in range(0, len(hist), batch):
        for h in hist[s:s + batch]:
            st, _ = planner._state(h)
            out.append(st.float().cpu())
    return torch.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None,
                    help="(state, action) cache; reuses the decoder cache")
    ap.add_argument("--state-cache", default=None)
    ap.add_argument("--n-problems", type=int, default=4000)
    ap.add_argument("--n-val-problems", type=int, default=300)
    ap.add_argument("--n-codes", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--context", action="store_true",
                    help="context-conditioned prior: cross-attend from the "
                         "state over the frozen context hidden states "
                         "(recomputed per batch from the cached history "
                         "token ids, as the decoder does). Default off = "
                         "the original pooled-state MLP prior.")
    ap.add_argument("--ctx-d-model", type=int, default=384)
    ap.add_argument("--ctx-layers", type=int, default=2)
    ap.add_argument("--ctx-heads", type=int, default=6)
    ap.add_argument("--max-ctx", type=int, default=768)
    ap.add_argument("--max-phrase", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    model, vocab, cfg = load_flat_run(args.ckpt, str(device))
    for p in model.parameters():
        p.requires_grad_(False)
    planner = FlatPlanner(model, vocab, device,
                          max_len=int(cfg["model"]["max_len"]))

    t0 = time.time()
    cache_path = Path(args.cache) if args.cache else None
    if cache_path is not None and cache_path.exists():
        blob = torch.load(cache_path, weights_only=False)
        (h_tr, a_tr, _, _), (h_va, a_va, _, _) = blob["tr"], blob["va"]
    else:
        tr_ds, _ = build_eval_dataset(cfg, vocab, args.n_problems,
                                      cfg["data"]["train_seed"])
        va_ds, _ = build_eval_dataset(cfg, vocab, args.n_val_problems, 2)
        h_tr, a_tr, _, _ = cache_pairs(planner, tr_ds, args.n_problems,
                                       vocab, args.max_phrase)
        h_va, a_va, _, _ = cache_pairs(planner, va_ds, args.n_val_problems,
                                       vocab, args.max_phrase)
        if cache_path is not None:
            torch.save({"tr": (h_tr, a_tr, [], []),
                        "va": (h_va, a_va, [], [])}, cache_path)

    sc = Path(args.state_cache) if args.state_cache else None
    if sc is not None and sc.exists():
        blob = torch.load(sc, weights_only=False)
        s_tr, s_va = blob["s_tr"], blob["s_va"]
    else:
        s_tr = states_for(planner, h_tr)
        s_va = states_for(planner, h_va)
        if sc is not None:
            torch.save({"s_tr": s_tr, "s_va": s_va}, sc)
    print(f"data ready in {time.time()-t0:.0f}s: {len(s_tr)} train, "
          f"{len(s_va)} val", flush=True)

    prior = CodePrior(CodePriorConfig(
        dim=s_tr.shape[1], n_codes=args.n_codes, hidden=args.hidden,
        n_layers=args.n_layers, residual=not args.no_residual,
        use_context=args.context, ctx_d_model=args.ctx_d_model,
        ctx_layers=args.ctx_layers, ctx_heads=args.ctx_heads))
    prior.fit_scaling(a_tr, s_tr)
    prior.init_codebook(a_tr, generator=torch.Generator().manual_seed(args.seed))
    prior = prior.to(device)
    a_tr, s_tr = a_tr.to(device), s_tr.to(device)
    a_va, s_va = a_va.to(device), s_va.to(device)

    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=0.01)
    N = len(s_tr)
    rng = random.Random(args.seed)
    log = []

    def ctx_for(hist_all, idx_list):
        """Frozen context hidden states for a batch, recomputed exactly as
        the decoder's training does (cached token ids -> model.encode)."""
        toks, mask, _ = pad_batch(
            [hist_all[i] for i in idx_list], [[] for _ in idx_list],
            vocab.pad_id, args.max_ctx, 1, device)
        return context_hiddens(model, toks, mask), mask

    for ep in range(args.epochs):
        prior.train()
        order = list(range(N))
        rng.shuffle(order)
        tot = {"loss": 0.0, "ce": 0.0, "recon": 0.0}
        nb = 0
        for s in range(0, N, args.batch_size):
            ii = order[s:s + args.batch_size]
            idx = torch.tensor(ii, device=device)
            prior.ema_update(a_tr[idx])
            if args.context:
                ctx, cmask = ctx_for(h_tr, ii)
                out = prior.loss(s_tr[idx], a_tr[idx], ctx=ctx,
                                 ctx_mask=cmask)
            else:
                out = prior.loss(s_tr[idx], a_tr[idx])
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            for k in tot:
                if k in out:
                    tot[k] += float(out[k])
            nb += 1
        prior.eval()
        with torch.no_grad():
            idx = prior.quantize(a_va)
            if args.context:
                lgs, ces = [], []
                for s in range(0, len(s_va), args.batch_size):
                    ii = list(range(s, min(s + args.batch_size, len(s_va))))
                    ctx, cmask = ctx_for(h_va, ii)
                    lgs.append(prior.logits(s_va[ii], ctx, cmask))
                lg = torch.cat(lgs)
                v = {"ce": nn_ce(lg, idx)}
            else:
                v = prior.loss(s_va, a_va)
                lg = prior.logits(s_va)
            top1 = float((lg.argmax(-1) == idx).float().mean())
            top4 = float((lg.topk(4, -1).indices == idx.unsqueeze(1))
                         .any(-1).float().mean())
            used = len(set(prior.quantize(a_tr).tolist()))
        row = {"epoch": ep, "train_loss": tot["loss"] / max(nb, 1),
               "train_ce": tot["ce"] / max(nb, 1),
               "val_ce": float(v["ce"]), "val_top1": top1, "val_top4": top4,
               "codes_used": used}
        log.append(row)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(json.dumps(row), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prior.cpu().save(str(out))
    Path(str(out) + ".json").write_text(json.dumps(
        {"ckpt": args.ckpt, "n_codes": args.n_codes,
         "residual": not args.no_residual, "context": bool(args.context),
         "history": log,
         "final": log[-1] if log else None}, indent=2))
    print(json.dumps(log[-1] if log else {}))


if __name__ == "__main__":
    main()
