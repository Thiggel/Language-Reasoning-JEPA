"""F_hi fidelity audit: is one macro jump as good as three chained F steps?

For states along held-out traces, sample up to 8 feasible 3-step action
sequences; predict the end state via (a) F_hi(s, macro(a_1..a_3)) and
(b) F(F(F(s,a_1),a_2),a_3); execute each sequence symbolically and encode
the true end states. Reports nearest-neighbor matching accuracy and mean
LN-L1 error for both routes.

Usage: python scripts/audit_hierarchy.py ckpt=runs/X/best.pt device=cuda:0
"""

from __future__ import annotations

import itertools
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


def _ln(x):
    return F.layer_norm(x, x.shape[-1:])


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    model, vocab, run_cfg = load_run(cfg.ckpt, cfg.device)
    dataset = build_dataset(run_cfg, vocab, split="val")
    device = torch.device(cfg.device)
    K = model.core.macro_k
    rng = random.Random(cfg.seed)

    def tokens(texts):
        ids = [vocab.encode(t) for t in texts]
        L = max(len(i) for i in ids)
        out = torch.full((1, len(ids), L), vocab.pad_id, dtype=torch.long)
        for c, i in enumerate(ids):
            out[0, c, : len(i)] = torch.tensor(i)
        return out.to(device)

    def enc_steps(pt, pm, texts):
        if not texts:
            empty = torch.full((1, 1, 1), vocab.pad_id, dtype=torch.long, device=device)
            return model.encode_states(pt, pm, empty, torch.zeros(1, 1, dtype=torch.bool, device=device))[0]
        st = tokens(texts)
        sm = torch.ones(1, st.shape[1], dtype=torch.bool, device=device)
        return model.encode_states(pt, pm, st, sm)[1][:, -1]

    stats = {"match_hi": [0, 0], "match_flat": [0, 0],
             "err_hi": [], "err_flat": [], "chance": []}
    n_eps = min(cfg.n_episodes, 60)
    with torch.no_grad():
        for ep in range(n_eps):
            problem, _ = dataset.problem(ep)
            env = SymbolicEnv(problem)
            prompt = prompt_sentences(problem, random.Random(cfg.seed + ep))
            pt = tokens(prompt)
            pm = torch.ones(1, pt.shape[1], dtype=torch.bool, device=device)
            step_texts: list[str] = []
            while not env.solved:
                # enumerate feasible K-step sequences from here
                seqs = []
                def dfs(e, prefix):
                    if len(prefix) == K:
                        seqs.append(prefix)
                        return
                    for a in e.feasible_actions():
                        c = e.clone()
                        c.step(a)
                        dfs(c, prefix + [a])
                dfs(env, [])
                rng.shuffle(seqs)
                seqs = seqs[:8]
                if len(seqs) >= 2:
                    s = enc_steps(pt, pm, step_texts)
                    n = len(seqs)
                    a = model.encode_actions(
                        tokens([action_phrase(problem, i) for q in seqs for i in q])
                        .squeeze(0).unsqueeze(1)
                    ).squeeze(1).reshape(n, K, -1)
                    macro = model.core.macro_encoder(a)
                    pred_hi = model.core.hi_predictor(s.expand(n, -1), macro)
                    cur = s.expand(n, -1)
                    for d in range(K):
                        cur = model.predictor(cur, a[:, d])
                    pred_flat = cur
                    true = []
                    for q in seqs:
                        c = env.clone()
                        texts = list(step_texts)
                        for i in q:
                            texts.append(c.step(i))
                        true.append(enc_steps(pt, pm, texts))
                    true = torch.cat(true)
                    for key, pred in (("hi", pred_hi), ("flat", pred_flat)):
                        d_mat = torch.cdist(_ln(pred), _ln(true), p=1)
                        stats[f"match_{key}"][0] += int(
                            (d_mat.argmin(1) == torch.arange(n, device=device)).sum()
                        )
                        stats[f"match_{key}"][1] += n
                        stats[f"err_{key}"].append(
                            (_ln(pred) - _ln(true)).abs().mean().item()
                        )
                    stats["chance"].append(1.0 / n)
                nec = [x for x in env.feasible_actions() if x in problem.query_ancestors]
                step_texts.append(env.step(min(nec)))
    out = {
        "match_hi": round(stats["match_hi"][0] / max(stats["match_hi"][1], 1), 4),
        "match_flat": round(stats["match_flat"][0] / max(stats["match_flat"][1], 1), 4),
        "err_hi": round(sum(stats["err_hi"]) / max(len(stats["err_hi"]), 1), 4),
        "err_flat": round(sum(stats["err_flat"]) / max(len(stats["err_flat"]), 1), 4),
        "chance": round(sum(stats["chance"]) / max(len(stats["chance"]), 1), 4),
        "n": stats["match_hi"][1],
    }
    print(json.dumps(out, indent=2))
    dest = Path(cfg.out or Path(cfg.ckpt).parent / "hierarchy_audit.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved to {dest}")


if __name__ == "__main__":
    main()
