"""Closed-loop latent planning on the FAITHFUL iGSM domain.

Same protocol as the stylized planner: enumerate every currently feasible
parameter, encode its intent phrase, score F(s, a) with the value head,
execute the argmin in the official environment, re-encode, and replan.
Lookahead greater than one uses the reference environment to enumerate future
feasible sequences and is therefore an explicitly opt-in diagnostic.
"""

from __future__ import annotations

import random

import torch

from textjepa.data.faithful import FaithfulDataset, FaithfulEnv
from textjepa.data.vocab import Vocab
from textjepa.planning.evaluate import aggregate_episodes
from textjepa.planning.search import EpisodeResult


class FaithfulPlanner:
    def __init__(self, model, vocab: Vocab, device: torch.device,
                 lookahead: int = 1, max_expand: int = 64,
                 allow_oracle_future_actions: bool = False):
        if lookahead > 1 and not allow_oracle_future_actions:
            raise ValueError(
                "lookahead > 1 enumerates future actions with the reference "
                "environment; set allow_oracle_future_actions=true only for "
                "a labeled oracle-action diagnostic"
            )
        self.model = model
        self.vocab = vocab
        self.device = device
        self.lookahead = lookahead
        self.max_expand = max_expand
        self.allow_oracle_future_actions = allow_oracle_future_actions

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

    def _sequences(
        self, env: FaithfulEnv, rng: random.Random
    ) -> list[list]:
        """Balanced fixed-depth rollouts with absorbing terminal padding."""
        roots = list(env.feasible_actions())
        rng.shuffle(roots)
        if not roots:
            return [[None] * self.lookahead]
        if self.lookahead == 1:
            return [[root] for root in roots]
        total = max(self.max_expand, len(roots))
        quotient, remainder = divmod(total, len(roots))
        sequences = []
        for root_index, root in enumerate(roots):
            for _ in range(quotient + int(root_index < remainder)):
                candidate = env.clone()
                candidate.resolved.append(root)  # feasibility-only transition
                sequence = [root]
                for _step in range(1, self.lookahead):
                    if candidate.solved:
                        sequence.append(None)
                        continue
                    feasible = list(candidate.feasible_actions())
                    if not feasible:
                        sequence.append(None)
                        continue
                    action = rng.choice(feasible)
                    candidate.resolved.append(action)
                    sequence.append(action)
                sequences.append(sequence)
        rng.shuffle(sequences)
        return sequences

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
            seqs = self._sequences(
                env, random.Random(f"{seed}:{len(step_texts)}:candidates")
            )
            n = len(seqs)
            depth = max(len(q) for q in seqs)
            cur = s.expand(n, -1).clone()
            # Constant across candidates; unlike the historical accumulated
            # path length, it cannot disclose which rollout solved early.
            cost = torch.full(
                (n,), float(self.lookahead), device=self.device
            )
            for d in range(depth):
                texts, alive = [], []
                for q in seqs:
                    # ``_sequences`` pads with None once a rollout absorbs
                    # (solved / no feasible action); those slots are dead.
                    entry = q[d] if d < len(q) else None
                    if entry is not None:
                        texts.append(env.action_text(entry))
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
            total = cost + self.model.value_head(cur, s0.expand(n, -1))
            best = seqs[int(total.argmin().item())]
            q = best[0]
            n_distr += int(q not in fp.necessary)
            step_texts.append(env.step(q))
        return EpisodeResult(
            env.solved, len(step_texts), len(fp.necessary), n_distr,
            solved_at=len(step_texts) if env.solved else None,
        )


def evaluate_faithful_planning(
    planner: FaithfulPlanner, dataset: FaithfulDataset, n_episodes: int,
    slack: int = 0, seed: int = 0, slack_curve: bool = False,
) -> dict[str, dict]:
    """Faithful iGSM planning metrics in the shared JSON shape.

    With ``slack_curve`` the episodes are run once at the generous budget
    ``necessary + slack`` and then scored at every smaller slack; the policy
    never reads its budget, so the run at slack ``s`` is a prefix of the run at
    the largest slack. The scalar metrics therefore describe the generous run,
    and ``success_by_slack[str(slack)]`` equals ``success``.
    """
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
        rand_.append(EpisodeResult(
            env.solved, steps, len(fp.necessary), n_d,
            solved_at=steps if env.solved else None,
        ))

    def agg(rs):
        return aggregate_episodes(rs, slack_curve=slack_curve, slack=slack)

    return {"latent_planner": agg(planned), "random_policy": agg(rand_)}
