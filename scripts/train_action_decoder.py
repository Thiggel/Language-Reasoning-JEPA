"""DECODER GATE: can an action VECTOR + the PROBLEM CONTEXT be rendered back
to the exact executable action phrase, including phrases never seen in
training?

    .venv/bin/python scripts/train_action_decoder.py \
        --ckpt <best.pt> --out dec.pt --arms full,no_action,no_context

Every previous catalogue-free proposer failed here (parse rate .000).  This
script measures whether conditioning the decoder on the prompt fixes it.
The backbone is FROZEN and every input is taken under ``no_grad``; the
decoder shares no parameter with it, so no config key is added and no
existing run changes.

Supervision is the DEMONSTRATED SOLUTION TRACE the data already carries: at
each state, the action that actually occurred, and its own text.  No
symbolic state, no steps-to-go, no feasibility labels.  (Symbolic state is
used only to WALK the demonstrated trace, exactly as
``scripts/train_flow_prior.py`` does.)

ARMS (same architecture, one input zeroed -- a capacity-matched control):
  full        action vector + context
  no_action   context only     -- how much is just "guess the next action"?
  no_context  action vector only -- reproduces the old decoder's information
  shuffled_action  action vector from a DIFFERENT state, context intact --
                   separates copying from actually reading the action code.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from textjepa.planning.action_decoder import ContextActionDecoder
from textjepa.planning.flat_search import FlatPlanner
from textjepa.utils import seed_everything

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_flat import build_eval_dataset, load_flat_run  # noqa: E402

ARMS = ("full", "no_action", "no_context", "shuffled_action")


@torch.no_grad()
def cache_pairs(planner: FlatPlanner, dataset, n_problems: int, vocab,
                max_phrase: int, offpath_prob: float = 0.0, seed: int = 0):
    """(history tokens, action code, phrase tokens) along solution traces.

    Identical trace walk to ``train_flow_prior.cache_pairs``.  Context hidden
    states are NOT cached (too large); they are recomputed in batches from the
    cached history token ids, which is exact.
    """
    # offpath_prob: probability of stepping onto a random feasible action
    # instead of the reference solution, so the decoder (and prior) see the
    # off-path states that planning actually visits. The pair label is still
    # just the action that was taken -- no symbolic supervision.
    rng = random.Random(seed)
    hist, codes, phrases, texts = [], [], [], []
    for i in range(n_problems):
        fp = dataset.problem(i)[0]
        env = fp.make_env()
        history = [t for s in fp.prompt_sentences for t in vocab.encode(s)]
        step_cap = 4 * max(1, len(fp.necessary))
        steps = 0
        while not env.solved and steps < step_cap:
            steps += 1
            feasible = env.feasible_actions()
            todo = [q for q in feasible if q in fp.necessary]
            if not todo:
                break
            if offpath_prob > 0.0 and rng.random() < offpath_prob:
                q = feasible[rng.randrange(len(feasible))]
            else:
                q = todo[0]
            text = env.action_text(q)
            toks = vocab.encode(text)
            if len(toks) + 1 > max_phrase:
                break
            code = planner._codes(history, [toks])[0]
            hist.append(list(history))
            codes.append(code.float().cpu())
            phrases.append(toks)
            texts.append(text)
            history = history + toks
            history = history + vocab.encode(env.step(q))
    return hist, torch.stack(codes), phrases, texts


def pad_batch(hist, phrases, pad_id, max_len, max_phrase, device):
    L = max(len(h) for h in hist)
    L = min(L, max_len)
    B = len(hist)
    toks = torch.full((B, L), pad_id, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.bool)
    for b, h in enumerate(hist):
        h = h[-max_len:]
        toks[b, :len(h)] = torch.tensor(h)
        mask[b, :len(h)] = False
    tgt = torch.full((B, max_phrase), pad_id, dtype=torch.long)
    for b, p in enumerate(phrases):
        tgt[b, :len(p)] = torch.tensor(p)  # PAD after the phrase == EOS
    return toks.to(device), mask.to(device), tgt.to(device)


@torch.no_grad()
def context_hiddens(model, toks, mask):
    h = model.encode(toks)
    return h.float().detach()


def run_arms(arms, cache_tr, cache_va, model, vocab, cfg, args, device,
             seen_texts):
    """Train every arm in ONE pass over the data.

    All arms consume the SAME frozen context hidden states, and that encoder
    forward dominates the cost, so sharing it makes the four-arm ablation
    roughly as cheap as a single arm -- and guarantees the arms see byte-
    identical inputs and batch order, which is what makes the comparison a
    controlled one.
    """
    hist_tr, codes_tr, ph_tr, _ = cache_tr
    hist_va, codes_va, ph_va, tx_va = cache_va
    d_state = int(cfg["model"]["d_model"])
    decs, opts, scheds = {}, {}, {}
    N = len(hist_tr)
    steps = args.epochs * ((N + args.batch_size - 1) // args.batch_size)
    for arm in arms:
        seed_everything(args.seed)  # identical init across arms
        dec = ContextActionDecoder(
            d_state, len(vocab), max_len=args.max_phrase,
            d_model=args.d_model, n_layers=args.n_layers,
            n_heads=args.n_heads, use_action=(arm != "no_action"),
            use_context=(arm != "no_context"),
        ).to(device)
        decs[arm] = dec
        opts[arm] = torch.optim.AdamW(dec.parameters(), lr=args.lr,
                                      weight_decay=0.01)
        scheds[arm] = torch.optim.lr_scheduler.OneCycleLR(
            opts[arm], args.lr, total_steps=max(steps, 1), pct_start=0.1)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    rng = random.Random(args.seed)
    logs = {a: [] for a in arms}
    for ep in range(args.epochs):
        for d in decs.values():
            d.train()
        order = list(range(N))
        rng.shuffle(order)
        tot = {a: 0.0 for a in arms}
        nb = 0
        for s in range(0, N, args.batch_size):
            idx = order[s:s + args.batch_size]
            toks, mask, tgt = pad_batch(
                [hist_tr[i] for i in idx], [ph_tr[i] for i in idx],
                vocab.pad_id, args.max_ctx, args.max_phrase, device)
            ctx = context_hiddens(model, toks, mask)
            base_act = codes_tr[torch.tensor(idx)].to(device)
            target = tgt.clone()
            for b, i in enumerate(idx):
                k = len(ph_tr[i])
                if k + 1 < args.max_phrase:
                    target[b, k + 1:] = -100
            perm = torch.randperm(len(idx), device=device)
            for arm in arms:
                act = base_act[perm] if arm == "shuffled_action" else base_act
                logits = decs[arm](act, ctx, mask, tgt)
                loss = lossf(logits.reshape(-1, logits.shape[-1]),
                             target.reshape(-1))
                opts[arm].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(decs[arm].parameters(), 1.0)
                opts[arm].step()
                scheds[arm].step()
                tot[arm] += float(loss)
            nb += 1
        for arm in arms:
            logs[arm].append({"epoch": ep, "loss": tot[arm] / max(nb, 1)})
        print(f"epoch {ep} " + " ".join(
            f"{a}={tot[a]/max(nb,1):.4f}" for a in arms), flush=True)

    # ------------------------------------------------------------- eval
    for d in decs.values():
        d.eval()
    stats = {a: dict(exact=0, es=0, eu=0) for a in arms}
    preds = {a: [] for a in arms}
    n_seen = n_unseen = 0
    for s in range(0, len(hist_va), args.batch_size):
        idx = list(range(s, min(s + args.batch_size, len(hist_va))))
        toks, mask, _ = pad_batch(
            [hist_va[i] for i in idx], [ph_va[i] for i in idx],
            vocab.pad_id, args.max_ctx, args.max_phrase, device)
        ctx = context_hiddens(model, toks, mask)
        base_act = codes_va[torch.tensor(idx)].to(device)
        perm = torch.randperm(len(idx), device=device)
        novel = [tx_va[i] not in seen_texts for i in idx]
        n_unseen += sum(novel)
        n_seen += len(idx) - sum(novel)
        for arm in arms:
            act = base_act[perm] if arm == "shuffled_action" else base_act
            out = decs[arm].generate(act, ctx, mask, eos_id=vocab.pad_id)
            for b, i in enumerate(idx):
                ok = out[b] == ph_va[i]
                stats[arm]["exact"] += int(ok)
                if novel[b]:
                    stats[arm]["eu"] += int(ok)
                else:
                    stats[arm]["es"] += int(ok)
                if len(preds[arm]) < 40:
                    preds[arm].append({"gold": tx_va[i],
                                       "pred": vocab.decode(out[b]),
                                       "novel": novel[b]})
    n = len(hist_va)
    results = {}
    for arm in arms:
        st = stats[arm]
        results[arm] = {
            "arm": arm, "n_val": n,
            "exact_match": st["exact"] / max(n, 1),
            "n_novel": n_unseen, "n_seen": n_seen,
            "exact_match_novel": st["eu"] / max(n_unseen, 1),
            "exact_match_seen": st["es"] / max(n_seen, 1),
            "train_loss": logs[arm][-1]["loss"] if logs[arm] else None,
            "history": logs[arm], "samples": preds[arm],
        }
    return results, decs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="full,no_action,no_context")
    ap.add_argument("--n-problems", type=int, default=4000)
    ap.add_argument("--n-val-problems", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--max-phrase", type=int, default=24)
    ap.add_argument("--max-ctx", type=int, default=768)
    ap.add_argument("--op-lo", type=int, default=None)
    ap.add_argument("--op-hi", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--offpath-prob", type=float, default=0.0)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    model, vocab, cfg = load_flat_run(args.ckpt, str(device))
    for p in model.parameters():
        p.requires_grad_(False)
    planner = FlatPlanner(model, vocab, device,
                          max_len=int(cfg["model"]["max_len"]))

    cache_path = Path(args.cache) if args.cache else None
    if cache_path is not None and cache_path.exists():
        blob = torch.load(cache_path, weights_only=False)
        cache_tr, cache_va = blob["tr"], blob["va"]
    else:
        t0 = time.time()
        tr_ds, _ = build_eval_dataset(cfg, vocab, args.n_problems,
                                      cfg["data"]["train_seed"])
        va_ds, prov = build_eval_dataset(cfg, vocab, args.n_val_problems, 2,
                                         op_lo=args.op_lo, op_hi=args.op_hi)
        cache_tr = cache_pairs(planner, tr_ds, args.n_problems, vocab,
                               args.max_phrase,
                               offpath_prob=args.offpath_prob, seed=args.seed)
        cache_va = cache_pairs(planner, va_ds, args.n_val_problems, vocab,
                               args.max_phrase,
                               offpath_prob=args.offpath_prob,
                               seed=args.seed + 1)
        print(f"cached {len(cache_tr[0])} train / {len(cache_va[0])} val "
              f"pairs in {time.time()-t0:.0f}s", flush=True)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"tr": cache_tr, "va": cache_va}, cache_path)

    seen_texts = set(cache_tr[3])
    novel = sum(1 for t in cache_va[3] if t not in seen_texts)
    print(f"val phrases never seen verbatim in training: "
          f"{novel}/{len(cache_va[3])}", flush=True)

    results = {"ckpt": args.ckpt, "n_train_pairs": len(cache_tr[0]),
               "n_val_pairs": len(cache_va[0]),
               "novel_val_fraction": novel / max(len(cache_va[3]), 1),
               "op_range": [args.op_lo, args.op_hi], "arms": {}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    for arm in arms:
        assert arm in ARMS, arm
    res, decs = run_arms(arms, cache_tr, cache_va, model, vocab, cfg, args,
                         device, seen_texts)
    results["arms"] = res
    for arm in arms:
        print(json.dumps({k: v for k, v in res[arm].items()
                          if k not in ("history", "samples")}), flush=True)
    if "full" in decs:
        decs["full"].save(str(out))
    Path(str(out) + ".json").write_text(json.dumps(results, indent=2))
    print(json.dumps({a: r["exact_match"] for a, r in results["arms"].items()}))


if __name__ == "__main__":
    main()
