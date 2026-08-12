"""Read-out controls for the failed contrastive action prior (CPU-only).

Question
--------
The contrastive action prior (softmax over the per-problem action catalogue,
trained on the frozen backbone's pooled state ``s_t`` with detached candidate
embeddings ``u(c)``) sits at chance, i.e. ``ln M``.  The LDAD cycle-consistency
route reaches feasibility AUC ~.94 from the *same* inputs.  Is the failure

  (A) information genuinely absent from ``s_t``,
  (B) present but the composition/binding step (does ``u(c)``'s named parent
      set sit inside ``s_t``'s resolved set?) is hard for a shallow head, or
  (C) mundane undertraining / undercapacity of the prior head?

Two controls, both offline against a frozen checkpoint; nothing in ``src/`` is
touched and no trainer is involved.

CONTROL 1 -- oracle-labelled read-out probes on ``(s_t, u(v))``:
  * resolvedness: is variable ``v`` already computed at step ``t``?
  * feasibility: is action ``c`` feasible at step ``t``?
  Probe families: logistic regression, 2-layer MLP (hidden 128), and -- for
  feasibility -- a bilinear form ``s^T W u``.  Baselines: majority class and
  a shuffled-state control (``s`` taken from an unrelated problem/step).

CONTROL 2 -- stronger contrastive heads replicating the prior's own task:
  per step, score every catalogue candidate and take softmax-CE against the
  observed action index.  Heads: (a) the original GaussianActionPrior, (b) a
  wide concat MLP, (c) bilinear + MLP hybrid, (d) a 2-layer cross-attention
  scorer where ``u(c)`` attends over the causal sequence of per-sentence
  hidden vectors available at step ``t``.

NOTE (evidence labelling): every probe/label here is CANDIDATE-PRIVILEGED
diagnostic evidence -- resolvedness and feasibility labels come from the
symbolic oracle.  Nothing here is a planning result.

Usage:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/probe_state_feasibility.py \
        --ckpt <path/best.pt> --n-train 2000 --n-val 500 --out results.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from textjepa.data.igsm.dataset import collate
from textjepa.data.igsm.render import catalogue_phrases
from textjepa.models.heads import GaussianActionPrior
from textjepa.utils.checkpoint import build_dataset, load_run

torch.set_num_threads(min(16, torch.get_num_threads()))


# --------------------------------------------------------------------------- #
# feature extraction (read-only use of the frozen model)
# --------------------------------------------------------------------------- #
def _pad_token_chunks(seqs: list[list[int]], pad: int) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    return torch.tensor(
        [s + [pad] * (width - len(s)) for s in seqs], dtype=torch.long
    )


def _full_hidden(model, batch: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Per-sentence causal hidden states h over [prompt | steps].

    Re-runs ``DiscourseStateModel.forward``'s body (read-only) so that the
    pre-pooling per-sentence vectors -- which the public API discards, it
    returns only ``s0`` and the per-step states -- are available to the
    cross-attention scorer of CONTROL 2 (d).
    """
    sm = model.state_model
    prompt_emb = model.encode_chunks(batch["prompt_tokens"])
    step_emb = model.encode_chunks(batch["step_tokens"])
    x = torch.cat(
        [prompt_emb + sm.segment[0], step_emb + sm.segment[1]], dim=1
    )
    x = x + sm._positions(x.shape[1])
    valid = torch.cat([batch["prompt_mask"], batch["step_mask"]], dim=1)
    from textjepa.models.layers import build_causal_attention_mask

    attn_mask = build_causal_attention_mask(valid, sm.n_heads)
    h = sm.norm(sm.encoder(x, mask=attn_mask))
    return h, prompt_emb.shape[1]


def var_depths(problem) -> list[int]:
    depth = [0] * len(problem.vars)
    for v in problem.vars:  # vars are topologically ordered (parents < idx)
        depth[v.idx] = (
            0 if v.is_leaf else 1 + max(depth[p] for p in v.parents)
        )
    return depth


@torch.no_grad()
def extract(model, dataset, vocab, n: int, batch_size: int = 32) -> list[dict]:
    """Per-problem frozen features + oracle structure."""
    records: list[dict] = []
    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        items = [dataset[i] for i in idxs]
        batch = collate(items, vocab.pad_id)
        s0, step_states = model.encode_states(
            batch["prompt_tokens"], batch["prompt_mask"],
            batch["step_tokens"], batch["step_mask"],
        )
        prev_states = torch.cat(
            [s0.unsqueeze(1), step_states[:, :-1]], dim=1
        )  # [B, T, D]: state BEFORE step t
        hidden, n_prompt = _full_hidden(model, batch)
        prompt_len = batch["prompt_mask"].sum(1)
        for b, index in enumerate(idxs):
            problem, _ = dataset.problem(index)
            phrases = catalogue_phrases(problem)
            tokens = _pad_token_chunks(
                [vocab.encode(p) for p in phrases], vocab.pad_id
            )
            u_small = model.encode_actions(tokens.unsqueeze(0))[0]  # [V, d_a]
            u_wide = model.encode_chunks(tokens.unsqueeze(0))[0]  # [V, D]
            trace = items[b]["var_idx"]
            T = len(trace)
            p_len = int(prompt_len[b])
            # causal sequence visible before step t is
            #   hidden[:p_len]  ++  hidden[n_prompt : n_prompt + t]
            seq = torch.cat(
                [hidden[b, :p_len], hidden[b, n_prompt:n_prompt + T]], dim=0
            )
            records.append(dict(
                s=prev_states[b, :T].clone(),
                seq=seq.clone(),
                p_len=p_len,
                u_small=u_small.clone(),
                u_wide=u_wide.clone(),
                trace=list(trace),
                parents=[tuple(v.parents) for v in problem.vars],
                depth=var_depths(problem),
                n_vars=len(problem.vars),
            ))
    return records


# --------------------------------------------------------------------------- #
# CONTROL 1: binary read-out probes
# --------------------------------------------------------------------------- #
def build_pair_rows(records: list[dict], u_key: str, onehot: bool = False):
    """Rows over (problem, step t, variable v)."""
    states, us, resolved, feasible, depth, vid = [], [], [], [], [], []
    max_vars = max(r["n_vars"] for r in records)
    for r in records:
        done: set[int] = set()
        for t in range(len(r["trace"])):
            s = r["s"][t]
            for v in range(r["n_vars"]):
                states.append(s)
                if onehot:
                    e = torch.zeros(max_vars)
                    e[v] = 1.0
                    us.append(e)
                else:
                    us.append(r[u_key][v])
                is_done = v in done
                resolved.append(float(is_done))
                feasible.append(float(
                    (not is_done)
                    and all(pa in done for pa in r["parents"][v])
                ))
                depth.append(r["depth"][v])
                vid.append(v)
            done.add(r["trace"][t])
    return (
        torch.stack(states), torch.stack(us),
        torch.tensor(resolved), torch.tensor(feasible),
        torch.tensor(depth), torch.tensor(vid),
    )


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    pos, neg = labels > 0.5, labels <= 0.5
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.double)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    # average ranks over ties
    vals, inverse, counts = scores.unique(
        return_inverse=True, return_counts=True
    )
    sums = torch.zeros(len(vals), dtype=torch.double).index_add_(
        0, inverse, ranks
    )
    ranks = (sums / counts)[inverse]
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def mlp_probe(d_in: int, hidden: int = 128) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, 1),
    )


class Bilinear(nn.Module):
    """s^T W u (+ linear terms + bias)."""

    def __init__(self, d_s: int, d_u: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(d_s, d_u))
        nn.init.normal_(self.W, std=0.02)
        self.lin_s = nn.Linear(d_s, 1)
        self.lin_u = nn.Linear(d_u, 1, bias=False)
        self.d_s = d_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, u = x[:, : self.d_s], x[:, self.d_s:]
        return (
            ((s @ self.W) * u).sum(-1, keepdim=True)
            + self.lin_s(s) + self.lin_u(u)
        )


def train_binary(model, Xtr, ytr, Xva, yva, epochs=40, bs=4096, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, patience = math.inf, None, 0
    n = len(Xtr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[j]).squeeze(-1), ytr[j])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(lossf(model(Xva).squeeze(-1), yva))
        if val < best - 1e-4:
            best, patience = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = model(Xva).squeeze(-1)
    return scores


def summarize(scores, labels, depth) -> dict:
    acc = float(((scores > 0).float() == labels).float().mean())
    out = {"acc": acc, "auc": auc(scores, labels),
           "pos_rate": float(labels.mean())}
    by_depth = {}
    for d in sorted(set(depth.tolist())):
        m = depth == d
        if int(m.sum()) < 50:
            continue
        by_depth[int(d)] = {
            "n": int(m.sum()),
            "acc": float(((scores[m] > 0).float() == labels[m]).float().mean()),
            "auc": auc(scores[m], labels[m]),
        }
    out["by_depth"] = by_depth
    return out


def standardize(Xtr, Xva):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-5)
    return (Xtr - mu) / sd, (Xva - mu) / sd


DEFAULT_FEATURE_SPECS = [
    ("u_action", dict(u_key="u_small", onehot=False)),
    ("onehot_var", dict(u_key="u_small", onehot=True)),
]


def run_control1(train_rec, val_rec, seed: int, feature_specs=None) -> dict:
    results: dict = {}
    for feat_name, kwargs in feature_specs or DEFAULT_FEATURE_SPECS:
        Str, Utr, Rtr, Ftr, Dtr, _ = build_pair_rows(train_rec, **kwargs)
        Sva, Uva, Rva, Fva, Dva, _ = build_pair_rows(val_rec, **kwargs)
        d_s, d_u = Str.shape[1], Utr.shape[1]
        Xtr = torch.cat([Str, Utr], 1)
        Xva = torch.cat([Sva, Uva], 1)
        Xtr, Xva = standardize(Xtr, Xva)
        # shuffled-state control: s from an unrelated row
        g = torch.Generator().manual_seed(seed)
        perm_tr = torch.randperm(len(Xtr), generator=g)
        perm_va = torch.randperm(len(Xva), generator=g)
        Xtr_sh = torch.cat([Xtr[perm_tr, :d_s], Xtr[:, d_s:]], 1)
        Xva_sh = torch.cat([Xva[perm_va, :d_s], Xva[:, d_s:]], 1)
        for task, ytr, yva in [
            ("resolved", Rtr, Rva), ("feasible", Ftr, Fva)
        ]:
            block = {
                "n_train_rows": len(Xtr), "n_val_rows": len(Xva),
                "majority_acc": float(max(yva.mean(), 1 - yva.mean())),
            }
            torch.manual_seed(seed)
            block["logreg"] = summarize(
                train_binary(nn.Linear(d_s + d_u, 1), Xtr, ytr, Xva, yva),
                yva, Dva,
            )
            torch.manual_seed(seed)
            block["mlp128"] = summarize(
                train_binary(mlp_probe(d_s + d_u), Xtr, ytr, Xva, yva),
                yva, Dva,
            )
            if task == "feasible":
                torch.manual_seed(seed)
                block["bilinear"] = summarize(
                    train_binary(Bilinear(d_s, d_u), Xtr, ytr, Xva, yva),
                    yva, Dva,
                )
            torch.manual_seed(seed)
            block["mlp128_shuffled_state"] = summarize(
                train_binary(mlp_probe(d_s + d_u), Xtr_sh, ytr, Xva_sh, yva),
                yva, Dva,
            )
            results[f"{feat_name}/{task}"] = block
    return results


# --------------------------------------------------------------------------- #
# CONTROL 2: catalogue-softmax heads
# --------------------------------------------------------------------------- #
class WideConcatMLP(nn.Module):
    def __init__(self, d_s: int, d_u: int, hidden: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(d_s)
        self.net = nn.Sequential(
            nn.Linear(d_s + d_u, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, u, seq=None, seq_mask=None):
        s = self.norm(s).unsqueeze(1).expand(-1, u.shape[1], -1)
        return self.net(torch.cat([s, u], -1)).squeeze(-1)


class BilinearHybrid(nn.Module):
    def __init__(self, d_s: int, d_u: int, hidden: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(d_s)
        self.W = nn.Parameter(torch.zeros(d_s, d_u))
        nn.init.normal_(self.W, std=0.02)
        self.net = nn.Sequential(
            nn.Linear(d_s + d_u, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, u, seq=None, seq_mask=None):
        s = self.norm(s)
        bil = torch.einsum("bd,de,bve->bv", s, self.W, u)
        s_e = s.unsqueeze(1).expand(-1, u.shape[1], -1)
        return bil + self.net(torch.cat([s_e, u], -1)).squeeze(-1)


class CrossAttentionScorer(nn.Module):
    """u(c) (as a query) attends over the causal per-sentence hidden states."""

    def __init__(self, d_s: int, d_u: int, d: int = 256, heads: int = 8):
        super().__init__()
        self.q = nn.Linear(d_u, d)
        self.kv_norm = nn.LayerNorm(d_s)
        self.s_proj = nn.Linear(d_s, d)
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                attn=nn.MultiheadAttention(d, heads, batch_first=True),
                n1=nn.LayerNorm(d), n2=nn.LayerNorm(d),
                ff=nn.Sequential(
                    nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
                ),
            )) for _ in range(2)
        ])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, s, u, seq, seq_mask):
        B, V, _ = u.shape
        kv = self.kv_norm(seq)
        kv = torch.cat([self.s_proj(kv), self.s_proj(self.kv_norm(s)).unsqueeze(1)], 1)
        mask = torch.cat(
            [~seq_mask, torch.zeros(B, 1, dtype=torch.bool, device=s.device)], 1
        )
        kv = kv.unsqueeze(1).expand(-1, V, -1, -1).reshape(B * V, -1, kv.shape[-1])
        mask = mask.unsqueeze(1).expand(-1, V, -1).reshape(B * V, -1)
        x = self.q(u).reshape(B * V, 1, -1)
        for blk in self.blocks:
            a, _ = blk["attn"](blk["n1"](x), kv, kv, key_padding_mask=mask)
            x = x + a
            x = x + blk["ff"](blk["n2"](x))
        return self.out(x).reshape(B, V)


class OriginalPriorHead(nn.Module):
    """The deployed head: a Gaussian action prior scored over the catalogue."""

    def __init__(self, d_s: int, d_u: int, hidden: int = 256):
        super().__init__()
        self.prior = GaussianActionPrior(d_s, d_u, hidden=hidden)

    def forward(self, s, u, seq=None, seq_mask=None):
        V = u.shape[1]
        return self.prior.log_prob(s.unsqueeze(1).expand(-1, V, -1), u)


def build_step_rows(records: list[dict], device="cpu"):
    max_vars = max(r["n_vars"] for r in records)
    max_len = max(r["seq"].shape[0] for r in records)
    d_s = records[0]["s"].shape[1]
    d_u = records[0]["u_small"].shape[1]
    S, U, M, Y, SEQ, SEQM, NF = [], [], [], [], [], [], []
    for r in records:
        done: set[int] = set()
        for t in range(len(r["trace"])):
            S.append(r["s"][t])
            u = torch.zeros(max_vars, d_u)
            u[: r["n_vars"]] = r["u_small"]
            U.append(u)
            m = torch.zeros(max_vars, dtype=torch.bool)
            m[: r["n_vars"]] = True
            M.append(m)
            Y.append(r["trace"][t])
            n_feas = sum(
                1 for v in range(r["n_vars"])
                if v not in done and all(pa in done for pa in r["parents"][v])
            )
            NF.append(n_feas)
            visible = r["p_len"] + t
            seq = torch.zeros(max_len, d_s)
            seq[:visible] = r["seq"][:visible]
            SEQ.append(seq)
            sm = torch.zeros(max_len, dtype=torch.bool)
            sm[:visible] = True
            SEQM.append(sm)
            done.add(r["trace"][t])
    return dict(
        s=torch.stack(S), u=torch.stack(U), mask=torch.stack(M),
        y=torch.tensor(Y), seq=torch.stack(SEQ), seq_mask=torch.stack(SEQM),
        n_feasible=torch.tensor(NF, dtype=torch.float),
    )


def feasible_labels(records: list[dict], max_vars: int) -> torch.Tensor:
    rows = []
    for r in records:
        done: set[int] = set()
        for t in range(len(r["trace"])):
            f = torch.zeros(max_vars)
            for v in range(r["n_vars"]):
                if v not in done and all(pa in done for pa in r["parents"][v]):
                    f[v] = 1.0
            rows.append(f)
            done.add(r["trace"][t])
    return torch.stack(rows)


def masked_ce(logits, mask, y):
    logits = logits.masked_fill(~mask, float("-inf"))
    return nn.functional.cross_entropy(logits, y)


def train_scorer(head, tr, va, epochs=60, bs=256, lr=1e-3, needs_seq=False):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n = len(tr["y"])
    best, best_state, patience = math.inf, None, 0
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            logits = head(
                tr["s"][j], tr["u"][j],
                tr["seq"][j] if needs_seq else None,
                tr["seq_mask"][j] if needs_seq else None,
            )
            loss = masked_ce(logits, tr["mask"][j], tr["y"][j])
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        head.eval()
        with torch.no_grad():
            val_logits = eval_logits(head, va, needs_seq)
            val = float(masked_ce(val_logits, va["mask"], va["y"]))
        if val < best - 1e-4:
            best, patience = val, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        return best, eval_logits(head, va, needs_seq)


def eval_logits(head, data, needs_seq, bs=512):
    outs = []
    for i in range(0, len(data["y"]), bs):
        sl = slice(i, i + bs)
        outs.append(head(
            data["s"][sl], data["u"][sl],
            data["seq"][sl] if needs_seq else None,
            data["seq_mask"][sl] if needs_seq else None,
        ))
    return torch.cat(outs)


def run_control2(train_rec, val_rec, seed: int) -> dict:
    tr, va = build_step_rows(train_rec), build_step_rows(val_rec)
    max_vars = tr["u"].shape[1]
    feas_va = feasible_labels(val_rec, max_vars)
    d_s, d_u = tr["s"].shape[1], tr["u"].shape[2]
    n_cand = va["mask"].sum(1).float()
    chance = float(n_cand.log().mean())
    floor = float(va["n_feasible"].clamp_min(1).log().mean())
    out = {
        "n_train_steps": int(len(tr["y"])), "n_val_steps": int(len(va["y"])),
        "chance_ce_ln_M": chance,
        "feasible_entropy_floor": floor,
        "mean_catalogue_size": float(n_cand.mean()),
        "mean_n_feasible": float(va["n_feasible"].mean()),
        "heads": {},
    }
    heads = [
        ("a_original_gaussian_prior", lambda: OriginalPriorHead(d_s, d_u), False),
        ("b_wide_mlp_512x3", lambda: WideConcatMLP(d_s, d_u), False),
        ("c_bilinear_mlp_hybrid", lambda: BilinearHybrid(d_s, d_u), False),
        ("d_cross_attention", lambda: CrossAttentionScorer(d_s, d_u), True),
    ]
    for name, ctor, needs_seq in heads:
        torch.manual_seed(seed)
        t0 = time.time()
        head = ctor()
        val_ce, logits = train_scorer(head, tr, va, needs_seq=needs_seq)
        masked = logits.masked_fill(~va["mask"], float("-inf"))
        top1 = float((masked.argmax(1) == va["y"]).float().mean())
        flat_mask = va["mask"].reshape(-1)
        out["heads"][name] = {
            "val_ce": val_ce,
            "top1": top1,
            "feasibility_auc": auc(
                logits.reshape(-1)[flat_mask].double(),
                feas_va.reshape(-1)[flat_mask],
            ),
            "params": sum(p.numel() for p in head.parameters()),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  head {name}: {out['heads'][name]}", flush=True)
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="probe_state_feasibility.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, vocab, cfg = load_run(args.ckpt, device="cpu")
    train_ds = build_dataset(cfg, vocab, split="train", size=args.n_train)
    val_ds = build_dataset(cfg, vocab, split="val", size=args.n_val)
    t0 = time.time()
    print("extracting frozen features ...", flush=True)
    train_rec = extract(model, train_ds, vocab, args.n_train)
    val_rec = extract(model, val_ds, vocab, args.n_val)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    results = {
        "ckpt": args.ckpt, "n_train": args.n_train, "n_val": args.n_val,
        "seed": args.seed, "d_state": train_rec[0]["s"].shape[1],
        "d_action": train_rec[0]["u_small"].shape[1],
    }
    print("CONTROL 1: read-out probes ...", flush=True)
    results["control1_probes"] = run_control1(train_rec, val_rec, args.seed)
    for key, block in results["control1_probes"].items():
        print(f"  {key}: " + ", ".join(
            f"{k}={v['auc']:.3f}/{v['acc']:.3f}"
            for k, v in block.items() if isinstance(v, dict)
        ), flush=True)
    print("CONTROL 2: contrastive heads ...", flush=True)
    results["control2_heads"] = run_control2(train_rec, val_rec, args.seed)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
