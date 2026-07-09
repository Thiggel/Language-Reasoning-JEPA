"""Closed-loop latent planning (MPC) over discourse actions.

The planner may query the *interface* of the world — which actions are
feasible (dependency preconditions are stated in the prompt) and their
intent phrases — but consequences and goal progress are judged purely in
latent space: candidate actions are rolled out with the predictor and
scored with the value head (estimated total steps = depth + predicted
remaining). The chosen action is executed by the symbolic environment,
whose outcome sentence is re-encoded before replanning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.graph import Problem
from textjepa.data.igsm.render import prompt_sentences
from textjepa.data.vocab import Vocab


@dataclass
class EpisodeResult:
    solved: bool
    steps: int
    n_necessary: int
    n_distractor: int


def _feasible(problem: Problem, resolved: frozenset[int]) -> list[int]:
    return [
        v.idx
        for v in problem.vars
        if v.idx not in resolved and all(p in resolved for p in v.parents)
    ]


def _sequences(
    problem: Problem, resolved: frozenset[int], depth: int, cap: int
) -> list[list[int]]:
    """All feasible action sequences up to ``depth`` (stop early if solved)."""
    seqs: list[list[int]] = []
    frontier: list[tuple[list[int], frozenset[int]]] = [([], resolved)]
    for _ in range(depth):
        nxt = []
        for seq, res in frontier:
            for a in _feasible(problem, res):
                new = (seq + [a], res | {a})
                if problem.query in new[1]:
                    seqs.append(new[0])
                else:
                    nxt.append(new)
        frontier = nxt[:cap]
        if not frontier:
            break
    seqs.extend(seq for seq, _ in frontier)
    return seqs[: cap * 4] if seqs else [[]]


class LatentPlanner:
    def __init__(
        self,
        model,
        vocab: Vocab,
        device: torch.device,
        lookahead: int = 1,
        max_expand: int = 64,
        energy: str = "value",  # "value" | "oracle_goal"
    ):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.lookahead = lookahead
        self.max_expand = max_expand
        self.energy = energy

    def _tokens(self, texts: list[str], min_chunks: int = 0) -> torch.Tensor:
        ids = [self.vocab.encode(t) for t in texts]
        C = max(len(ids), min_chunks, 1)
        L = max((len(i) for i in ids), default=1)
        out = torch.full((1, C, L), self.vocab.pad_id, dtype=torch.long)
        for c, i in enumerate(ids):
            out[0, c, : len(i)] = torch.tensor(i)
        return out.to(self.device)

    @torch.no_grad()
    def plan_episode(self, problem: Problem, slack: int = 0, seed: int = 0) -> EpisodeResult:
        env = SymbolicEnv(problem)
        prompt = prompt_sentences(problem, random.Random(seed))
        prompt_tokens = self._tokens(prompt)
        prompt_mask = torch.ones(1, len(prompt), dtype=torch.bool, device=self.device)
        step_texts: list[str] = []
        budget = problem.n_necessary_steps + slack
        n_distractor = 0
        goal_state = (
            self._oracle_goal_state(problem, prompt_tokens, prompt_mask)
            if self.energy == "oracle_goal"
            else None
        )

        while not env.solved and len(step_texts) < budget:
            s = self._current_state(prompt_tokens, prompt_mask, step_texts)
            s0 = self._s0(prompt_tokens, prompt_mask)
            seqs = _sequences(
                problem, frozenset(env.resolved_set), self.lookahead, self.max_expand
            )
            best = self._best_sequence(s, s0, problem, seqs, goal_state)
            chosen = best[0]
            n_distractor += int(chosen not in problem.query_ancestors)
            step_texts.append(env.step(chosen))

        return EpisodeResult(
            env.solved, len(step_texts), problem.n_necessary_steps, n_distractor
        )

    def _s0(self, prompt_tokens, prompt_mask) -> torch.Tensor:
        if not hasattr(self, "_s0_cache") or self._s0_cache[0] is not prompt_tokens:
            empty = torch.full(
                (1, 1, 1), self.vocab.pad_id, dtype=torch.long, device=self.device
            )
            no_steps = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
            s0, _ = self.model.encode_states(
                prompt_tokens, prompt_mask, empty, no_steps
            )
            self._s0_cache = (prompt_tokens, s0)
        return self._s0_cache[1]

    def _current_state(self, prompt_tokens, prompt_mask, step_texts) -> torch.Tensor:
        if not step_texts:
            return self._s0(prompt_tokens, prompt_mask)
        return self._encode_steps(prompt_tokens, prompt_mask, step_texts)

    @torch.no_grad()
    def _oracle_goal_state(
        self, problem: Problem, prompt_tokens, prompt_mask
    ) -> torch.Tensor:
        """Encode the solved terminal state (LeWM-style oracle diagnostic).

        Uses the ground-truth minimal solution, so this energy is an upper
        bound / distillation target, not a deployable planner.
        """
        env = SymbolicEnv(problem)
        texts = []
        while not env.solved:
            necessary = [
                a for a in env.feasible_actions() if a in problem.query_ancestors
            ]
            texts.append(env.step(min(necessary)))
        return self._encode_steps(prompt_tokens, prompt_mask, texts)

    def _encode_steps(self, prompt_tokens, prompt_mask, step_texts) -> torch.Tensor:
        step_tokens = self._tokens(step_texts)
        step_mask = torch.ones(
            1, len(step_texts), dtype=torch.bool, device=self.device
        )
        _, states = self.model.encode_states(
            prompt_tokens, prompt_mask, step_tokens, step_mask
        )
        return states[:, -1]

    def _best_sequence(
        self,
        s: torch.Tensor,
        s0: torch.Tensor,
        problem: Problem,
        seqs: list[list[int]],
        goal_state: torch.Tensor | None = None,
    ) -> list[int]:
        from textjepa.data.igsm.render import action_phrase

        depth = max(len(q) for q in seqs)
        n = len(seqs)
        cur = s.expand(n, -1).clone()
        cost = torch.zeros(n, device=self.device)
        for d in range(depth):
            idxs = [q[d] if d < len(q) else None for q in seqs]
            texts = [
                action_phrase(problem, i) if i is not None else "." for i in idxs
            ]
            tokens = self._tokens(texts).squeeze(0).unsqueeze(1)  # [n, 1, L]
            a = self.model.encode_actions(tokens).squeeze(1)
            step_alive = torch.tensor(
                [i is not None for i in idxs], device=self.device
            )
            nxt = self.model.predictor(cur, a)
            cur = torch.where(step_alive.unsqueeze(1), nxt, cur)
            cost = cost + step_alive.float()
        if goal_state is not None:
            ln = lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:])
            total = (ln(cur) - ln(goal_state)).abs().mean(-1)
        else:
            total = cost + self.model.value_head(cur, s0.expand(n, -1))
        return seqs[int(total.argmin().item())]
