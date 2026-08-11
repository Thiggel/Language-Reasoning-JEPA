"""Slack-curve latent planning on the compiled observed-action domains.

Same protocol as the stylized iGSM planner in :mod:`textjepa.planning.search`:
enumerate the actions the interface exposes, encode each candidate intent
phrase, roll it out with the JEPA predictor, score the imagined endpoint with
the model's Energy, execute the argmin in the symbolic executor, re-encode the
observed outcome, and replan. The only differences are that states and actions
are plain text (there is no symbolic ``Problem``) and that the executor comes
from :func:`textjepa.planning.catalogue.environment_from_episode`.

Candidate interfaces:

``feasible_menu``
    The executor's current legal-action menu. This is the headline,
    information-matched interface: every policy sees the same menu.
``full_catalogue``
    The episode's whole action catalogue. Infeasible proposals are executed
    with no-op semantics — the executor returns an "invalid" outcome sentence,
    the hidden state is unchanged, and the attempt is counted in
    ``invalid_action_rate`` — matching ``invalid_action_mode=noop`` in the
    stylized domain.

Other interfaces (latent-descent cycles, CEM over action codes) are developed
elsewhere; they plug in by overriding :meth:`ObservedActionPlanner._candidates`
and/or :meth:`ObservedActionPlanner.choose`, which are the only two places
that decide what gets scored.
"""

from __future__ import annotations

import os
import random

import torch

from textjepa.data.observed_action import ObservedActionEpisode
from textjepa.planning.catalogue import (
    CatalogueLatentPlanner,
    TextActionEnvironment,
    environment_from_episode,
)
from textjepa.planning.evaluate import aggregate_episodes
from textjepa.planning.search import (
    EpisodeResult,
    controlled_argmin,
    endpoint_energy,
)

# Planner-side interface name -> executor-side candidate interface name.
CANDIDATE_INTERFACES = {
    "feasible_menu": "feasible_menu",
    "full_catalogue": "full",
}


def _result(
    environment: TextActionEnvironment, steps: int, solved_at: int | None
) -> EpisodeResult:
    """Wrap an executor outcome in the shared planning-metrics record.

    ``n_necessary`` is the reference plan length (the domain's optimal /
    expert length), and ``n_distractor`` counts executed actions beyond it, so
    ``distractor_rate`` reads as an excess-action rate. The external domains
    have no per-action necessity label, so no stronger notion is claimed.
    """
    optimal = int(environment.optimal_length)
    return EpisodeResult(
        bool(environment.solved),
        steps,
        optimal,
        max(steps - optimal, 0),
        int(environment.invalid_actions),
        solved_at,
    )


class ObservedActionPlanner(CatalogueLatentPlanner):
    """Energy-ranked closed-loop planner over text actions.

    Inherits the text/vocab encoding, action encoding, and latent rollout
    utilities of :class:`CatalogueLatentPlanner`, and scores imagined
    endpoints with :func:`textjepa.planning.search.endpoint_energy`, the same
    function the stylized iGSM planner uses. No proposal, feasibility, or
    prior head is consulted.
    """

    def __init__(
        self,
        model,
        vocab,
        device: torch.device,
        *,
        lookahead: int = 1,
        max_expand: int = 64,
        energy: str = "value",
        candidate_interface: str = "feasible_menu",
        invalid_action_mode: str = "noop",
        score_control: str = "model",
    ):
        if candidate_interface not in CANDIDATE_INTERFACES:
            raise ValueError(
                f"unknown candidate interface: {candidate_interface!r}"
            )
        if energy not in {"value", "oracle_goal"}:
            raise ValueError(f"unknown energy: {energy}")
        if energy == "oracle_goal":
            raise ValueError(
                "the observed-action domains have no encodable solved "
                "terminal state; energy=oracle_goal is unavailable here"
            )
        if invalid_action_mode not in {"noop", "failure"}:
            raise ValueError(
                f"unknown invalid action mode: {invalid_action_mode!r}"
            )
        if score_control not in {"model", "shuffle", "zero"}:
            raise ValueError(f"unknown score control: {score_control!r}")
        if lookahead > 1 and candidate_interface == "feasible_menu":
            raise ValueError(
                "lookahead > 1 on the feasible menu would need the executor "
                "to be cloned and rolled forward, which the external "
                "environments do not support; use "
                "candidate_interface=full_catalogue for deeper search"
            )
        super().__init__(
            model, vocab, device,
            simulation_depth=lookahead,
            proposal_top_m=1,
            beam_width=max_expand,
        )
        self.lookahead = int(lookahead)
        self.max_expand = int(max_expand)
        self.energy = energy
        self.candidate_interface = candidate_interface
        self.invalid_action_mode = invalid_action_mode
        self.score_control = score_control

    def _candidates(
        self, environment: TextActionEnvironment
    ) -> tuple[str, ...]:
        """Actions offered to the scorer at the current step."""
        return tuple(environment.catalogue)

    @torch.no_grad()
    def choose(
        self, prompt, outcomes, actions, catalogue, score_seed: str = "0"
    ) -> str:
        if not catalogue:
            raise ValueError("planning needs at least one candidate action")
        s0, state_history, action_history = self._observed_history(
            prompt, outcomes, actions
        )
        codes = self._action_codes(tuple(catalogue))
        sequences = [(index,) for index in range(len(catalogue))]
        cost = None
        for depth in range(1, self.lookahead + 1):
            if depth > 1:
                sequences = [
                    sequence + (index,)
                    for sequence in sequences
                    for index in range(len(catalogue))
                ]
            future = torch.stack([
                codes[list(sequence)] for sequence in sequences
            ])
            leaf = self._rollout(state_history, action_history, future)
            # Constant across candidates: the executed path length must never
            # disclose which imagined rollout reached the goal early.
            steps = torch.full(
                (len(sequences),), float(self.lookahead), device=self.device
            )
            cost = endpoint_energy(self.model, leaf, s0, steps, self.energy)
            if depth < self.lookahead:
                keep = torch.argsort(cost, stable=True)[
                    :self.max_expand
                ].tolist()
                sequences = [sequences[index] for index in keep]
        selected = controlled_argmin(cost, score_seed, self.score_control)
        return catalogue[sequences[selected][0]]

    @torch.no_grad()
    def plan_episode(
        self,
        environment: TextActionEnvironment,
        slack: int = 0,
        seed: int = 0,
    ) -> EpisodeResult:
        """Run one closed-loop episode under ``optimal + slack`` actions.

        The budget is enforced by this harness; the policy is never told how
        many actions remain.
        """
        environment.set_candidate_interface(
            CANDIDATE_INTERFACES[self.candidate_interface]
        )
        environment.set_candidate_order_seed(f"{seed}:candidates")
        budget = int(environment.optimal_length) + int(slack)
        outcomes: list[str] = []
        actions: list[str] = []
        while not environment.solved and len(actions) < budget:
            invalid_before = environment.invalid_actions
            action = self.choose(
                environment.prompt,
                outcomes,
                actions,
                self._candidates(environment),
                score_seed=f"{seed}:{len(actions)}:scores",
            )
            outcomes.append(environment.step(action))
            actions.append(action)
            if (
                self.invalid_action_mode == "failure"
                and environment.invalid_actions > invalid_before
            ):
                break
        solved_at = len(actions) if environment.solved else None
        return _result(environment, len(actions), solved_at)


class RandomMenuPolicy:
    """Uniform choice from whatever the shared candidate interface offers."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def plan_episode(self, environment, slack: int = 0, seed: int = 0):
        budget = int(environment.optimal_length) + int(slack)
        steps = 0
        while not environment.solved and steps < budget:
            environment.step(self.rng.choice(list(environment.catalogue)))
            steps += 1
        return _result(
            environment, steps, steps if environment.solved else None
        )


class FirstCandidatePolicy:
    """Ordering control: always take the first offered candidate."""

    def plan_episode(self, environment, slack: int = 0, seed: int = 0):
        budget = int(environment.optimal_length) + int(slack)
        steps = 0
        while not environment.solved and steps < budget:
            environment.step(environment.catalogue[0])
            steps += 1
        return _result(
            environment, steps, steps if environment.solved else None
        )


class ExpertReplayPolicy:
    """Privileged upper bound: replays the stored expert trajectory."""

    def plan_episode(self, environment, slack: int = 0, seed: int = 0):
        budget = int(environment.optimal_length) + int(slack)
        steps = 0
        for action in environment.expert_actions:
            if environment.solved or steps >= budget:
                break
            environment.step(action)
            steps += 1
        return _result(
            environment, steps, steps if environment.solved else None
        )


def _episodes(dataset, n_episodes: int) -> list[ObservedActionEpisode]:
    episodes = getattr(dataset, "episodes", None)
    if not episodes:
        raise ValueError(
            "observed-action planning requires a dataset exposing .episodes"
        )
    return list(episodes[: int(n_episodes)])


def _run_policy(
    policy, episodes, slack: int, seed: int, candidate_interface: str
) -> list[EpisodeResult]:
    """Run one policy over every episode on a shared candidate interface."""
    results = []
    for index, episode in enumerate(episodes):
        environment = environment_from_episode(episode)
        try:
            environment.set_candidate_interface(
                CANDIDATE_INTERFACES[candidate_interface]
            )
            environment.set_candidate_order_seed(f"{seed + index}:candidates")
            results.append(
                policy.plan_episode(
                    environment, slack=slack, seed=seed + index
                )
            )
        finally:
            close = getattr(environment, "close", None)
            if close is not None:
                close()
    return results


def evaluate_observed_action_planning(
    model,
    dataset,
    vocab,
    device: torch.device,
    *,
    n_episodes: int,
    slack: int = 0,
    slack_curve: bool = True,
    candidate_interface: str = "feasible_menu",
    lookahead: int = 1,
    max_expand: int = 64,
    energy: str = "value",
    invalid_action_mode: str = "noop",
    score_control: str = "model",
    seed: int = 0,
) -> dict[str, dict]:
    """Planner + reference policies on compiled observed-action episodes.

    Returns the long-standing planning-metrics JSON shape (see
    :func:`textjepa.planning.evaluate.evaluate_planning`): one entry per
    policy, each with ``success``, ``mean_steps``, ``mean_necessary``,
    ``distractor_rate`` and ``invalid_action_rate``, plus ``success_by_slack``
    and ``excess_steps`` when ``slack_curve`` is set. With ``slack_curve`` the
    scalar metrics describe the single generous-budget run at ``slack``, and
    ``success_by_slack[str(slack)]`` equals ``success``.

    ``oracle`` is privileged expert replay, not a deployable policy.
    """
    episodes = _episodes(dataset, n_episodes)
    if any(
        episode.domain == "alfworld-textworld" for episode in episodes
    ) and not os.environ.get("ALFWORLD_DATA"):
        raise ValueError(
            "interactive ALFWorld evaluation requires ALFWORLD_DATA"
        )
    planner = ObservedActionPlanner(
        model, vocab, device,
        lookahead=lookahead,
        max_expand=max_expand,
        energy=energy,
        candidate_interface=candidate_interface,
        invalid_action_mode=invalid_action_mode,
        score_control=score_control,
    )
    policies = {
        (
            "latent_planner" if energy == "value"
            else f"latent_planner_{energy}"
        ): planner,
        "random_policy": RandomMenuPolicy(seed),
        "first_feasible_policy": FirstCandidatePolicy(),
        "oracle": ExpertReplayPolicy(),
    }
    results = {}
    for name, policy in policies.items():
        episode_results = _run_policy(
            policy, episodes, slack, seed, candidate_interface
        )
        results[name] = aggregate_episodes(
            episode_results, slack_curve=slack_curve, slack=slack
        )
    return results
