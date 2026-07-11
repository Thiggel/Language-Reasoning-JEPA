"""Closed-loop latent planning on the FAITHFUL iGSM domain.

Same protocol as the stylized planner: enumerate feasible parameters,
encode their intent phrases, score F(s, a) with the value head, execute
the argmin in the official environment, re-encode, replan. Lookahead-d
enumerates feasible d-step sequences (capped).
"""

from __future__ import annotations

import random

import torch

from textjepa.data.faithful import FaithfulDataset, FaithfulEnv
from textjepa.data.vocab import Vocab
from textjepa.planning.search import EpisodeResult


class FaithfulPlanner:
    def __init__(self, model, vocab: Vocab, device: torch.device,
                 lookahead: int = 1, max_expand: int = 64):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.lookahead = lookahead
        self.max_expand = max_expand

    def _tokens(self, texts: list[str]) -> torch.Tensor:
        ids = [self.vocab.encode(t) for t in texts]
        L = max(len(i) for i in ids)
        out = torch.full((1, len(ids), L), self.vocab.pad_id, dtype=torch.long)
        for c, i in enumerate(ids):
            out[0, c, : len(i)] = torch.tensor(i)
        return out.to(self.device)

    def _state(self, pt, pm, step_texts):
        if not step_texts:
            empty = torch.full((1, 1, 1), self.vocab.pad_id, dtype=torch.long,
                               device=self.device)
            return self.model.encode_states(
                pt, pm, empty,
                torch.zeros(1, 1, dtype=torch.bool, device=self.device),
            )[0]
        st = self._tokens(step_texts)
        sm = torch.ones(1, st.shape[1], dtype=torch.bool, device=self.device)
        return self.model.encode_states(pt, pm, st, sm)[1][:, -1]

    def _sequences(self, env: FaithfulEnv) -> list[list]:
        seqs, frontier = [], [([], env)]
        for _ in range(self.lookahead):
            nxt = []
            for prefix, e in frontier:
                for q in e.feasible_actions():
                    c = e.clone()
                    c.resolved.append(q)  # feasibility-only step (no render)
                    nxt.append((prefix + [q], c))
                    if len(nxt) >= self.max_expand:
                        break
                if len(nxt) >= self.max_expand:
                    break
            frontier = nxt
            seqs += [p for p, _ in frontier]
        # keep deepest-first unique prefixes, cap
        seqs = sorted(seqs, key=len, reverse=True)[: self.max_expand * 4]
        return seqs or [[]]

    @torch.no_grad()
    def plan_episode(self, fp, slack: int = 0, seed: int = 0) -> EpisodeResult:
        env = FaithfulEnv(fp)
        pt = self._tokens(fp.prompt_sentences)
        pm = torch.ones(1, pt.shape[1], dtype=torch.bool, device=self.device)
        step_texts: list[str] = []
        budget = len(fp.necessary) + slack
        n_distr = 0
        s0 = self._state(pt, pm, [])
        while not env.solved and len(step_texts) < budget:
            s = self._state(pt, pm, step_texts) if step_texts else s0
            seqs = self._sequences(env)
            n = len(seqs)
            depth = max(len(q) for q in seqs)
            cur = s.expand(n, -1).clone()
            cost = torch.zeros(n, device=self.device)
            for d in range(depth):
                texts, alive = [], []
                for q in seqs:
                    if d < len(q):
                        texts.append(env.action_text(q[d]))
                        alive.append(True)
                    else:
                        texts.append(".")
                        alive.append(False)
                a = self.model.encode_actions(
                    self._tokens(texts).squeeze(0).unsqueeze(1)
                ).squeeze(1)
                alive_t = torch.tensor(alive, device=self.device)
                nxt = self.model.predictor(cur, a)
                cur = torch.where(alive_t.unsqueeze(1), nxt, cur)
                cost = cost + alive_t.float()
            total = cost + self.model.value_head(cur, s0.expand(n, -1))
            best = seqs[int(total.argmin().item())]
            q = best[0]
            n_distr += int(q not in fp.necessary)
            step_texts.append(env.step(q))
        return EpisodeResult(
            env.solved, len(step_texts), len(fp.necessary), n_distr
        )


def evaluate_faithful_planning(
    planner: FaithfulPlanner, dataset: FaithfulDataset, n_episodes: int,
    slack: int = 0, seed: int = 0,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    planned, rand_ = [], []
    for i in range(n_episodes):
        fp, _ = dataset.problem(i)
        planned.append(planner.plan_episode(fp, slack=slack, seed=seed + i))
        env = FaithfulEnv(fp)
        steps = n_d = 0
        budget = len(fp.necessary) + slack
        while not env.solved and steps < budget:
            q = rng.choice(env.feasible_actions())
            n_d += int(q not in fp.necessary)
            env.step(q)
            steps += 1
        rand_.append(EpisodeResult(env.solved, steps, len(fp.necessary), n_d))

    def agg(rs):
        n = len(rs)
        return {
            "success": sum(r.solved for r in rs) / n,
            "mean_steps": sum(r.steps for r in rs) / n,
            "mean_necessary": sum(r.n_necessary for r in rs) / n,
            "distractor_rate": sum(r.n_distractor for r in rs)
            / max(sum(r.steps for r in rs), 1),
        }

    return {"latent_planner": agg(planned), "random_policy": agg(rand_)}
