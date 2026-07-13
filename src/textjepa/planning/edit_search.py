"""Closed-loop latent planning over buffer edits: edit until perfect.

Candidates come from the edit interface (including harmful ones); their
consequences are judged only via the latent predictor + value head.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from textjepa.data.edits.trajectory import EditEnv, topo_necessary
from textjepa.data.vocab import Vocab


@dataclass
class EditEpisodeResult:
    solved: bool
    steps: int
    n_defects_initial: int


class EditPlanner:
    def __init__(
        self, model, vocab: Vocab, device: torch.device, energy: str = "value"
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.energy = energy

    def _chunk_tensor(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.vocab.encode(t) for t in texts]
        C = max(len(ids), 1)
        L = max((len(i) for i in ids), default=1)
        out = torch.full((1, C, L), self.vocab.pad_id, dtype=torch.long)
        mask = torch.zeros(1, C, dtype=torch.bool)
        for c, i in enumerate(ids):
            out[0, c, : len(i)] = torch.tensor(i)
            mask[0, c] = True
        return out.to(self.device), mask.to(self.device)

    def _buffer_state(self, prompt_t, prompt_m, sentences: list[str]) -> torch.Tensor:
        buf_t, buf_m = self._chunk_tensor(sentences or ["."])
        if not sentences:
            buf_m = torch.zeros_like(buf_m)
        states = self.model.encode_buffers(
            prompt_t, prompt_m, buf_t.unsqueeze(1), buf_m.unsqueeze(1)
        )
        return states[:, 0]

    def _oracle_goal_state(
        self, env: EditEnv, prompt_t, prompt_m
    ) -> torch.Tensor:
        """Encode the fully repaired buffer (LeWM-style oracle diagnostic):
        every necessary variable, true text, topological order."""
        texts = [env._true_text(i) for i in topo_necessary(env.p)]
        return self._buffer_state(prompt_t, prompt_m, texts)

    @torch.no_grad()
    def plan_episode(
        self, env: EditEnv, prompt: list[str], rng: random.Random, slack: int = 0
    ) -> EditEpisodeResult:
        prompt_t, prompt_m = self._chunk_tensor(prompt)
        n_initial = env.n_defects()
        budget = n_initial + slack
        s0 = self._buffer_state(prompt_t, prompt_m, env.sentences())
        goal = (
            self._oracle_goal_state(env, prompt_t, prompt_m)
            if self.energy == "oracle_goal"
            else None
        )
        steps = 0
        while not env.solved and steps < budget:
            s = (
                s0
                if steps == 0
                else self._buffer_state(prompt_t, prompt_m, env.sentences())
            )
            cands = env.candidate_edits(include_harmful=True, rng=rng)
            texts = [env.intent_text(e) for e in cands]
            tok, _ = self._chunk_tensor(texts)
            a = self.model.encode_actions(tok.squeeze(0).unsqueeze(1)).squeeze(1)
            n = a.shape[0]
            attn = getattr(self.model, "attn_pred", None)
            if attn is not None:
                bt, _ = self._chunk_tensor(env.sentences() or ["."])
                sent = self.model.encode_chunks(bt)
                sm = torch.ones(
                    1, sent.shape[1], dtype=torch.bool, device=self.device
                )
                s_next = attn(
                    sent.expand(n, -1, -1), sm.expand(n, -1),
                    s.expand(n, -1), a,
                )
            else:
                s_next = self.model.predictor(s.expand(n, -1), a)
            if goal is not None:
                geo = getattr(self.model.core, "geo_head", None)
                fin, g = (
                    (geo(s_next), geo(goal)) if geo is not None else (s_next, goal)
                )
                ln = lambda x: F.layer_norm(x, x.shape[-1:])
                score = (ln(fin) - ln(g.expand(n, -1))).abs().mean(-1)
            else:
                score = self.model.value_head(s_next, s0.expand(n, -1))
            env.apply(cands[int(score.argmin().item())])
            steps += 1
        return EditEpisodeResult(env.solved, steps, n_initial)


def random_edit_episode(env: EditEnv, rng: random.Random, slack: int) -> EditEpisodeResult:
    n_initial = env.n_defects()
    budget = n_initial + slack
    steps = 0
    while not env.solved and steps < budget:
        env.apply(rng.choice(env.candidate_edits(include_harmful=True, rng=rng)))
        steps += 1
    return EditEpisodeResult(env.solved, steps, n_initial)


def oracle_edit_episode(env: EditEnv, rng: random.Random) -> EditEpisodeResult:
    n_initial = env.n_defects()
    steps = 0
    while not env.solved:
        env.apply(rng.choice(env.fixing_edits()))
        steps += 1
    return EditEpisodeResult(True, steps, n_initial)


def evaluate_edit_planning(
    planner: EditPlanner, dataset, n_episodes: int, slack: int = 0, seed: int = 0
) -> dict[str, dict[str, float]]:
    def agg(rs: list[EditEpisodeResult]) -> dict[str, float]:
        n = len(rs)
        return {
            "success": sum(r.solved for r in rs) / n,
            "mean_steps": sum(r.steps for r in rs) / n,
            "mean_defects": sum(r.n_defects_initial for r in rs) / n,
        }

    rng = random.Random(seed)
    planned, rand_, oracle = [], [], []
    for i in range(n_episodes):
        env, prng, prompt = dataset.make_env(i)
        planned.append(planner.plan_episode(env, prompt, prng, slack=slack))
        env2, _, _ = dataset.make_env(i)
        rand_.append(random_edit_episode(env2, rng, slack))
        env3, _, _ = dataset.make_env(i)
        oracle.append(oracle_edit_episode(env3, rng))
    key = (
        "latent_planner"
        if planner.energy == "value"
        else f"latent_planner_{planner.energy}"
    )
    return {
        key: agg(planned),
        "random_policy": agg(rand_),
        "oracle": agg(oracle),
    }
