"""Non-oracle catalogue planning for compiled observed-action domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from textjepa.data.observed_action import ObservedActionEpisode
from textjepa.data.planbench import (
    BlocksAction,
    action_catalogue,
    goal_reached,
    is_applicable,
    render_state,
    transition,
)
from textjepa.data.proofwriter import (
    ProofRule,
    RuleApplication,
    render_fact,
    rule_applications,
)


class TextActionEnvironment(Protocol):
    prompt: tuple[str, ...]
    goal: str
    catalogue: tuple[str, ...]
    optimal_length: int
    invalid_actions: int

    @property
    def solved(self) -> bool: ...
    def step(self, action: str) -> str: ...


class BlocksworldEnvironment:
    def __init__(self, episode: ObservedActionEpisode):
        spec = episode.metadata["environment_spec"]
        self.prompt = episode.prompt
        self.goal = episode.goal
        self.catalogue = tuple(dict.fromkeys(
            action for value in episode.transitions for action in value.catalogue
        ))
        self.optimal_length = int(episode.metadata["optimal_plan_length"])
        self.invalid_actions = 0
        self.state = frozenset(tuple(atom) for atom in spec["initial"])
        self.goal_state = frozenset(tuple(atom) for atom in spec["goal"])
        actions = action_catalogue(tuple(spec["objects"]))
        self.actions = {action.text: action for action in actions}

    @property
    def solved(self) -> bool:
        return goal_reached(self.state, self.goal_state)

    def step(self, action: str) -> str:
        operation = self.actions.get(action)
        if operation is None or not is_applicable(self.state, operation):
            self.invalid_actions += 1
            return "The proposed action is invalid and the state is unchanged ."
        self.state = transition(self.state, operation)
        return render_state(self.state)


class ProofWriterEnvironment:
    def __init__(self, episode: ObservedActionEpisode):
        spec = episode.metadata["environment_spec"]
        self.prompt = episode.prompt
        self.goal = episode.goal
        self.catalogue = tuple(dict.fromkeys(
            action for value in episode.transitions for action in value.catalogue
        ))
        self.optimal_length = int(episode.metadata["optimal_derivation_length"])
        self.invalid_actions = 0
        self.state = frozenset(tuple(fact) for fact in spec["initial"])
        self.target = tuple(spec["target"])
        self.rules = tuple(ProofRule(
            str(item["rule_id"]),
            str(item["text"]),
            tuple(tuple(fact) for fact in item["antecedents"]),
            tuple(item["conclusion"]),
        ) for item in spec["rules"])

    @property
    def solved(self) -> bool:
        return self.target in self.state

    def step(self, action: str) -> str:
        available = {
            application.text: application
            for application in rule_applications(self.state, self.rules)
        }
        application = available.get(action)
        if application is None:
            self.invalid_actions += 1
            return "The proposed inference is invalid and no fact is added ."
        self.state = self.state | {application.conclusion}
        return render_fact(application.conclusion)


class FaithfulIGSMEnvironment:
    """Full-catalogue wrapper around the official iGSM executor."""

    def __init__(self, problem):
        from textjepa.data.faithful import FaithfulEnv

        self.problem = problem
        self.environment = FaithfulEnv(problem)
        self.prompt = tuple(problem.prompt_sentences)
        self.goal = problem.prompt_sentences[-1]
        self.catalogue = tuple(
            self.environment.action_text(action)
            for action in problem.action_order
        )
        self.actions = dict(zip(self.catalogue, problem.action_order))
        self.optimal_length = len(problem.necessary)
        self.invalid_actions = 0

    @property
    def solved(self) -> bool:
        return self.environment.solved

    def step(self, action: str) -> str:
        symbolic = self.actions.get(action)
        if (
            symbolic is None
            or symbolic not in self.environment.feasible_actions()
        ):
            self.invalid_actions += 1
            return "The proposed definition is invalid and nothing changes ."
        return self.environment.step(symbolic)


def environment_from_episode(
    episode: ObservedActionEpisode,
) -> TextActionEnvironment:
    if episode.domain == "planbench-blocksworld":
        return BlocksworldEnvironment(episode)
    if episode.domain == "proofwriter":
        return ProofWriterEnvironment(episode)
    raise ValueError(
        f"domain {episode.domain!r} requires an interactive evaluator"
    )


def environment_from_faithful_problem(problem) -> FaithfulIGSMEnvironment:
    return FaithfulIGSMEnvironment(problem)


@dataclass
class CatalogueEpisodeResult:
    solved: bool
    steps: int
    optimal_length: int
    invalid_actions: int


class CatalogueLatentPlanner:
    """Propose from a learned full catalogue, then rerank latent rollouts."""

    def __init__(
        self,
        model,
        vocab,
        device: torch.device,
        simulation_depth: int = 1,
        proposal_top_m: int = 4,
        beam_width: int = 4,
        prior_only: bool = False,
        prior_weight: float = 0.0,
    ):
        if simulation_depth < 1 or proposal_top_m < 1 or beam_width < 1:
            raise ValueError("depth, proposal_top_m, and beam_width must be positive")
        self.model = model
        self.vocab = vocab
        self.device = device
        self.simulation_depth = int(simulation_depth)
        self.proposal_top_m = int(proposal_top_m)
        self.beam_width = int(beam_width)
        self.prior_only = bool(prior_only)
        self.prior_weight = float(prior_weight)

    def _chunks(self, texts: tuple[str, ...] | list[str]) -> torch.Tensor:
        encoded = [self.vocab.encode(text) for text in texts]
        count, width = max(len(encoded), 1), max(
            (len(value) for value in encoded), default=1
        )
        result = torch.full(
            (1, count, width), self.vocab.pad_id, dtype=torch.long,
            device=self.device,
        )
        for index, value in enumerate(encoded):
            result[0, index, :len(value)] = torch.tensor(
                value, device=self.device
            )
        return result

    def _action_codes(self, catalogue: tuple[str, ...]) -> torch.Tensor:
        tokens = self._chunks(catalogue).squeeze(0).unsqueeze(1)
        return self.model.encode_actions(tokens).squeeze(1)

    def _observed_history(self, prompt, outcomes, actions):
        prompt_tokens = self._chunks(prompt)
        prompt_mask = torch.ones(
            1, len(prompt), dtype=torch.bool, device=self.device
        )
        empty = torch.full(
            (1, 1, 1), self.vocab.pad_id, dtype=torch.long,
            device=self.device,
        )
        if outcomes:
            step_tokens = self._chunks(outcomes)
            step_mask = torch.ones(
                1, len(outcomes), dtype=torch.bool, device=self.device
            )
        else:
            step_tokens = empty
            step_mask = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        s0, states = self.model.encode_states(
            prompt_tokens, prompt_mask, step_tokens, step_mask
        )
        state_history = (
            torch.cat([s0.unsqueeze(1), states[:, :len(outcomes)]], 1)
            if outcomes else s0.unsqueeze(1)
        )
        action_history = (
            self._action_codes(tuple(actions)).unsqueeze(0)
            if actions
            else s0.new_zeros(1, 0, self.model.core.d_action)
        )
        return s0, state_history, action_history

    def _rollout(self, state_history, action_history, future_codes):
        start = state_history[:, -1]
        predictor = self.model.predictor
        if hasattr(predictor, "rollout"):
            return predictor.rollout(
                start.expand(future_codes.shape[0], -1),
                future_codes,
                state_history=state_history.expand(
                    future_codes.shape[0], -1, -1
                ),
                action_history=action_history.expand(
                    future_codes.shape[0], -1, -1
                ),
            )[:, -1]
        current = start.expand(future_codes.shape[0], -1)
        for index in range(future_codes.shape[1]):
            current = predictor(current, future_codes[:, index])
        return current

    @torch.no_grad()
    def choose(self, prompt, outcomes, actions, catalogue) -> str:
        s0, state_history, action_history = self._observed_history(
            prompt, outcomes, actions
        )
        codes = self._action_codes(catalogue)
        sequences: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        for depth in range(self.simulation_depth):
            expanded = []
            for sequence, cumulative_prior in sequences:
                if sequence:
                    future = codes[list(sequence)].unsqueeze(0)
                    leaf = self._rollout(
                        state_history, action_history, future
                    )
                else:
                    leaf = state_history[:, -1]
                support = self.model.core.action_support_head(
                    leaf.expand(len(catalogue), -1), codes
                )
                top = support.topk(min(self.proposal_top_m, len(catalogue))).indices
                for index in top.tolist():
                    expanded.append((
                        sequence + (index,),
                        cumulative_prior + float(support[index].item()),
                    ))
            sequences = sorted(
                expanded, key=lambda value: value[1], reverse=True
            )[:self.beam_width]
        if self.prior_only:
            return catalogue[sequences[0][0][0]]
        future = torch.stack([
            codes[list(sequence)] for sequence, _ in sequences
        ])
        leaf = self._rollout(state_history, action_history, future)
        energy = self.model.value_head(leaf, s0.expand(len(sequences), -1))
        if self.prior_weight:
            prior = energy.new_tensor([value for _, value in sequences])
            prior = (prior - prior.mean()) / prior.std(unbiased=False).clamp_min(1e-6)
            normalized_energy = (
                energy - energy.mean()
            ) / energy.std(unbiased=False).clamp_min(1e-6)
            score = normalized_energy - self.prior_weight * prior
        else:
            score = energy
        selected = sequences[int(score.argmin().item())][0][0]
        return catalogue[selected]

    @torch.no_grad()
    def run_episode(
        self, environment: TextActionEnvironment, excess_actions: int = 0
    ) -> CatalogueEpisodeResult:
        outcomes: list[str] = []
        actions: list[str] = []
        budget = environment.optimal_length + int(excess_actions)
        while not environment.solved and len(actions) < budget:
            action = self.choose(
                environment.prompt,
                outcomes,
                actions,
                environment.catalogue,
            )
            outcomes.append(environment.step(action))
            actions.append(action)
        return CatalogueEpisodeResult(
            environment.solved,
            len(actions),
            environment.optimal_length,
            environment.invalid_actions,
        )
