"""Counterfactual audit: does F(s, a) rank and place actions correctly?

For each visited state of held-out episodes we enumerate ALL feasible
actions (not just the executed one), predict their next-state latents,
and check them against the symbolic ground truth the model never saw:

1. ranking accuracy   — fraction of candidate pairs whose energy order
                        matches the true post-action remaining-steps order
                        (per energy: value head V(F(s,a), s0) and latent
                        goal distance), plus a top-1 "best action" hit rate
                        and Kendall's tau.
2. matching accuracy  — execute every candidate symbolically, encode the
                        true next states, and nearest-neighbor-match the
                        predictions to them (chance = 1/n_candidates).
3. RSA                — Pearson correlation between predicted and true
                        pairwise next-state distance matrices.

Usage: python scripts/audit_counterfactual.py ckpt=runs/X/best.pt device=cuda:0
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import action_phrase, prompt_sentences
from textjepa.utils.checkpoint import build_dataset, load_run
from textjepa.utils.seed import seed_everything


def _ln(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, x.shape[-1:])


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (_ln(a) - _ln(b)).abs().mean(-1)


def _pairs_correct(energy: torch.Tensor, true_rem: torch.Tensor) -> tuple[int, int]:
    """Count candidate pairs with strictly different true quality where the
    energy orders them correctly."""
    n = len(energy)
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if true_rem[i] == true_rem[j]:
                continue
            total += 1
            want = true_rem[i] < true_rem[j]  # i strictly better
            got = energy[i] < energy[j]
            correct += int(want == got)
    return correct, total


def _kendall(energy: torch.Tensor, true_rem: torch.Tensor) -> tuple[float, int]:
    n = len(energy)
    num = den = 0
    for i in range(n):
        for j in range(i + 1, n):
            dt = torch.sign(true_rem[i] - true_rem[j]).item()
            if dt == 0:
                continue
            de = torch.sign(energy[i] - energy[j]).item()
            num += dt * de
            den += 1
    return (num / den if den else 0.0), den


class _Tokenizer:
    def __init__(self, vocab, device):
        self.vocab, self.device = vocab, device

    def __call__(self, texts: list[str]) -> torch.Tensor:
        ids = [self.vocab.encode(t) for t in texts]
        L = max(len(i) for i in ids)
        out = torch.full((1, len(ids), L), self.vocab.pad_id, dtype=torch.long)
        for c, i in enumerate(ids):
            out[0, c, : len(i)] = torch.tensor(i)
        return out.to(self.device)


@torch.no_grad()
def audit_discourse(model, vocab, dataset, device, n_episodes: int, seed: int) -> dict:
    tok = _Tokenizer(vocab, device)
    geo = getattr(model.core, "geo_head", None)
    stats = {
        "pairs_value": [0, 0], "pairs_goal": [0, 0],
        "top1_value": [0, 0], "top1_goal": [0, 0],
        "tau_value": [], "tau_goal": [],
        "match": [0, 0], "rsa": [],
        "chance": [],
    }
    for ep in range(n_episodes):
        problem, _ = dataset.problem(ep)
        faithful = hasattr(problem, "prompt_sentences")
        if faithful:
            from textjepa.data.faithful import FaithfulEnv

            env = FaithfulEnv(problem)
            prompt = problem.prompt_sentences
            necessary = problem.necessary
            action_text = env.action_text
        else:
            env = SymbolicEnv(problem)
            prompt = prompt_sentences(problem, random.Random(seed + ep))
            necessary = problem.query_ancestors
            action_text = lambda action: action_phrase(problem, action)
        pt = tok(prompt)
        pm = torch.ones(1, len(pt[0]), dtype=torch.bool, device=device)
        step_texts: list[str] = []
        empty = torch.full(
            (1, 1, 1), vocab.pad_id, dtype=torch.long, device=device
        )
        s0 = model.encode_states(
            pt, pm, empty,
            torch.zeros(1, 1, dtype=torch.bool, device=device),
        )[0]
        # oracle goal latent for the goal-distance energy
        genv = FaithfulEnv(problem) if faithful else SymbolicEnv(problem)
        gtexts = []
        while not genv.solved:
            nec = [a for a in genv.feasible_actions() if a in necessary]
            # Official action identifiers are structured tuples.  Selecting
            # the first action in the reference environment's stable order
            # avoids imposing an artificial cross-type ordering.
            gtexts.append(genv.step(nec[0]))
        goal = _enc_steps(model, pt, pm, gtexts, tok, device)

        while not env.solved:
            feas = env.feasible_actions()
            if len(feas) >= 2:
                s = (
                    _enc_steps(model, pt, pm, step_texts, tok, device)
                    if step_texts
                    else s0
                )
                texts = [action_text(a) for a in feas]
                a_emb = model.encode_actions(
                    tok(texts).squeeze(0).unsqueeze(1)
                ).squeeze(1)
                n = len(feas)
                preds = model.predictor(s.expand(n, -1), a_emb)
                done = env.resolved_set
                true_rem = torch.tensor(
                    [len(necessary - (done | {a})) for a in feas],
                    dtype=torch.float,
                )
                e_val = model.value_head(preds, s0.expand(n, -1)).cpu()
                pg, gg = (geo(preds), geo(goal)) if geo is not None else (preds, goal)
                e_goal = _dist(pg, gg.expand(n, -1)).cpu()
                for key, e in (("value", e_val), ("goal", e_goal)):
                    c, t = _pairs_correct(e, true_rem)
                    stats[f"pairs_{key}"][0] += c
                    stats[f"pairs_{key}"][1] += t
                    tau, den = _kendall(e, true_rem)
                    if den:
                        stats[f"tau_{key}"].append(tau)
                    best = true_rem == true_rem.min()
                    stats[f"top1_{key}"][0] += int(best[e.argmin()].item())
                    stats[f"top1_{key}"][1] += 1
                # matching + RSA against symbolically executed next states
                nxt_texts = []
                for a in feas:
                    c = env.clone()
                    nxt_texts.append(step_texts + [c.step(a)])
                true_next = torch.cat(
                    [_enc_steps(model, pt, pm, t, tok, device) for t in nxt_texts]
                )
                d = torch.cdist(_ln(preds), _ln(true_next), p=1) / preds.shape[-1]
                stats["match"][0] += int((d.argmin(1) == torch.arange(n, device=device)).sum())
                stats["match"][1] += n
                stats["chance"].append(1.0 / n)
                if n >= 3:
                    dp = torch.pdist(_ln(preds), p=1)
                    dt_ = torch.pdist(_ln(true_next), p=1)
                    dp, dt_ = dp - dp.mean(), dt_ - dt_.mean()
                    denom = dp.norm() * dt_.norm()
                    if denom > 1e-8:
                        stats["rsa"].append((dp * dt_).sum().item() / denom.item())
            # follow the oracle to visit informative states
            nec = [a for a in env.feasible_actions() if a in necessary]
            step_texts.append(env.step(nec[0]))
    return _summarize(stats)


def _enc_steps(model, pt, pm, texts, tok, device):
    st = tok(texts)
    sm = torch.ones(1, st.shape[1], dtype=torch.bool, device=device)
    _, states = model.encode_states(pt, pm, st, sm)
    return states[:, -1]


@torch.no_grad()
def audit_edit(model, vocab, dataset, device, n_episodes: int, seed: int) -> dict:
    from textjepa.data.edits.trajectory import topo_necessary

    tok = _Tokenizer(vocab, device)
    geo = getattr(model.core, "geo_head", None)
    stats = {
        "pairs_value": [0, 0], "pairs_goal": [0, 0],
        "top1_value": [0, 0], "top1_goal": [0, 0],
        "tau_value": [], "tau_goal": [],
        "match": [0, 0], "rsa": [], "chance": [],
    }
    rng = random.Random(seed)

    def buf_state(prompt_t, prompt_m, sentences):
        bt = tok(sentences or ["."])
        bm = torch.ones(1, bt.shape[1], dtype=torch.bool, device=device)
        if not sentences:
            bm = torch.zeros_like(bm)
        return model.encode_buffers(
            prompt_t, prompt_m, bt.unsqueeze(1), bm.unsqueeze(1)
        )[:, 0]

    for ep in range(n_episodes):
        env, prng, prompt = dataset.make_env(ep)
        pt = tok(prompt)
        pm = torch.ones(1, pt.shape[1], dtype=torch.bool, device=device)
        goal_texts = [env._true_text(i) for i in topo_necessary(env.p)]
        goal = buf_state(pt, pm, goal_texts)
        s0 = buf_state(pt, pm, env.sentences())
        while not env.solved:
            cands = env.candidate_edits(include_harmful=True, rng=prng)
            if len(cands) >= 2:
                s = buf_state(pt, pm, env.sentences())
                texts = [env.intent_text(e) for e in cands]
                a = model.encode_actions(tok(texts).squeeze(0).unsqueeze(1)).squeeze(1)
                n = len(cands)
                attn = getattr(model, "attn_pred", None)
                if attn is not None:
                    bt = tok(env.sentences() or ["."])
                    sent = model.encode_chunks(bt)
                    sm = torch.ones(1, sent.shape[1], dtype=torch.bool,
                                    device=device)
                    preds = attn(sent.expand(n, -1, -1), sm.expand(n, -1),
                                 s.expand(n, -1), a)
                else:
                    preds = model.predictor(s.expand(n, -1), a)
                clones = []
                for e in cands:
                    c = env.clone()
                    c.apply(e)
                    clones.append(c)
                true_rem = torch.tensor(
                    [c.n_defects() for c in clones], dtype=torch.float
                )
                e_val = model.value_head(preds, s0.expand(n, -1)).cpu()
                pg, gg = (geo(preds), geo(goal)) if geo is not None else (preds, goal)
                e_goal = _dist(pg, gg.expand(n, -1)).cpu()
                for key, e in (("value", e_val), ("goal", e_goal)):
                    c_, t = _pairs_correct(e, true_rem)
                    stats[f"pairs_{key}"][0] += c_
                    stats[f"pairs_{key}"][1] += t
                    tau, den = _kendall(e, true_rem)
                    if den:
                        stats[f"tau_{key}"].append(tau)
                    best = true_rem == true_rem.min()
                    stats[f"top1_{key}"][0] += int(best[e.argmin()].item())
                    stats[f"top1_{key}"][1] += 1
                true_next = torch.cat(
                    [buf_state(pt, pm, c.sentences()) for c in clones]
                )
                d = torch.cdist(_ln(preds), _ln(true_next), p=1) / preds.shape[-1]
                stats["match"][0] += int((d.argmin(1) == torch.arange(n, device=device)).sum())
                stats["match"][1] += n
                stats["chance"].append(1.0 / n)
                if n >= 3:
                    dp = torch.pdist(_ln(preds), p=1)
                    dt_ = torch.pdist(_ln(true_next), p=1)
                    dp, dt_ = dp - dp.mean(), dt_ - dt_.mean()
                    denom = dp.norm() * dt_.norm()
                    if denom > 1e-8:
                        stats["rsa"].append((dp * dt_).sum().item() / denom.item())
            env.apply(rng.choice(env.fixing_edits()))
    return _summarize(stats)


def _summarize(stats: dict) -> dict:
    out = {}
    for key in ("pairs_value", "pairs_goal", "top1_value", "top1_goal", "match"):
        c, t = stats[key]
        out[key] = round(c / max(t, 1), 4)
        out[f"{key}_n"] = t
    for key in ("tau_value", "tau_goal", "rsa", "chance"):
        vals = stats[key]
        out[key] = round(sum(vals) / max(len(vals), 1), 4)
    return out


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    model, vocab, run_cfg = load_run(cfg.ckpt, cfg.device)
    dataset = build_dataset(run_cfg, vocab, split="val")
    device = torch.device(cfg.device)
    n = min(cfg.n_episodes, 100)
    if run_cfg.data.get("name", "igsm") == "igsm_edit":
        results = audit_edit(model, vocab, dataset, device, n, cfg.seed)
    else:
        results = audit_discourse(model, vocab, dataset, device, n, cfg.seed)
    for k, v in results.items():
        print(f"{k:16s} {v}")
    out = Path(cfg.out or Path(cfg.ckpt).parent / "counterfactual_audit.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
