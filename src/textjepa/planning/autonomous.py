"""AUTONOMOUS self-rollout: menu-free, oracle-free, end-to-end solutions.

This is the "usable model" interface of the master plan.  Per episode the
system does everything itself:

1. encode the problem prompt -> ``s_0``;
2. propose actions with NO menu, from the checkpoint's own action codebook
   (in-model ``action_codebook`` buffer when the checkpoint carries one, else
   the eval-time k-means codebook), grounded environment-side by nearest
   neighbour -- exactly the ``codebook_ground`` contract;
3. plan with the existing endpoint-Energy beam search over imagined states
   (:class:`~textjepa.planning.search.LatentPlanner`);
4. execute the chosen first action WITHOUT the oracle executor: the detached
   frozen-state sentence decoder
   (:class:`~textjepa.models.state_decoder.FrozenStateSentenceDecoder`) renders
   the next step's text from the IMAGINED state ``predictor(s, u)``;
5. append that self-generated text to the model's own context, re-encode, and
   repeat until the model claims completion (it writes a step sentence about
   the queried variable) or the step budget is exhausted.

The ground truth is touched ONLY to score: the final answer, and the per-step
diagnostics that localize where the self-rollout breaks (well-formedness,
whether the emitted sentence is about the action that was chosen, whether its
number is right, and the first step at which it diverges).  Nothing in the
loop reads feasibility, outcomes, or the reference solution.

Everything is frozen and deterministic: the backbone and the decoder are in
eval mode with ``requires_grad=False`` (asserted), the decoder samples greedily
and the codebook proposal is deterministic, so an episode is a pure function of
``(checkpoint, decoder, problem, seed)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random

import torch

from textjepa.data.igsm.graph import Problem
from textjepa.data.igsm.render import prompt_sentences
from textjepa.planning.evaluate import slack_curve_metrics
from textjepa.planning.search import EpisodeResult, LatentPlanner

Tensor = torch.Tensor

_STEP_PREFIX = ["so", "the", "number", "of"]


def parse_step_sentence(
    problem: Problem, text: str
) -> tuple[int | None, int | None]:
    """Inverse of :func:`textjepa.data.igsm.render.step_sentence`.

    Returns ``(variable index, stated value)``; either entry is ``None`` when
    the text does not denote that part of a well-formed step sentence of this
    problem.  A sentence is well formed when it names a variable of the problem
    and states an integer result -- the number after ``=`` for a computation,
    the number after ``is`` for a leaf lookup.
    """
    words = text.split()
    while words and words[-1] == ".":
        words = words[:-1]
    if words[: len(_STEP_PREFIX)] != _STEP_PREFIX or "is" not in words:
        return None, None
    at = words.index("is")
    name = " ".join(words[len(_STEP_PREFIX):at])
    names = {v.name: v.idx for v in problem.vars}
    idx = names.get(name)
    rest = words[at + 1:]
    value: int | None = None
    if "=" in rest:
        after = rest[rest.index("=") + 1:]
        token = after[0] if after else ""
    else:
        token = rest[0] if len(rest) == 1 else ""
    if token.lstrip("-").isdigit():
        value = int(token)
    return idx, value


def load_state_decoder(path: str, vocab, device) -> "object":
    """Rebuild a trained :class:`FrozenStateSentenceDecoder` from its file.

    ``scripts/train_state_decoder.py`` stores the shape metadata alongside the
    weights, so the decoder is reconstructed exactly as it was trained and no
    architecture arguments have to be repeated at eval time.
    """
    from textjepa.models.state_decoder import FrozenStateSentenceDecoder

    blob = torch.load(path, map_location=device, weights_only=False)
    decoder = FrozenStateSentenceDecoder(
        d_state=int(blob["d_state"]),
        vocab_size=len(vocab),
        max_len=int(blob["max_len"]),
        n_answers=int(blob.get("n_answers", 0)),
    )
    decoder.load_state_dict(blob["decoder"])
    return decoder.to(device).eval()


@dataclass
class AutonomousEpisode:
    """One self-rollout, with the diagnostics that localize its failure."""

    answer_correct: bool
    claimed_completion: bool
    answer: int | None
    true_answer: int
    n_steps: int
    n_necessary: int
    n_well_formed: int
    n_action_match: int
    n_value_correct: int
    first_divergence: int | None
    stalled: bool
    answer_head_correct: bool | None = None
    texts: list[str] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)


class AutonomousRollout(LatentPlanner):
    """Planner that renders its own next step instead of calling the env.

    Inherits every planning mechanism from :class:`LatentPlanner` (state
    encoding, causal history, endpoint-Energy beam search) and always uses the
    ``codebook_ground`` proposal interface; the only override is the execution
    step, which the frozen state decoder performs.
    """

    def __init__(
        self, model, vocab, device, decoder, stop_on_claim: bool = True,
        **kwargs,
    ):
        self.stop_on_claim = bool(stop_on_claim)
        kwargs.setdefault("candidate_interface", "codebook_ground")
        if kwargs["candidate_interface"] != "codebook_ground":
            raise ValueError(
                "the autonomous interface proposes with codebook_ground"
            )
        super().__init__(model, vocab, device, **kwargs)
        if self.energy != "value":
            raise ValueError(
                "the autonomous interface is oracle-free; energy must be "
                "'value' (oracle_goal/symbolic_distance read the environment)"
            )
        if self.simulator != "latent":
            raise ValueError("the autonomous interface imagines in latent space")
        self.decoder = decoder.to(device).eval()
        # JEPA-purity / frozen-eval assertion: nothing here can train, and no
        # gradient can reach the backbone through the read-out.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        assert not any(p.requires_grad for p in self.model.parameters()), (
            "backbone parameters must be frozen at autonomous eval time"
        )
        assert not any(p.requires_grad for p in self.decoder.parameters()), (
            "the state decoder is a detached read-out and must be frozen"
        )
        self.model.eval()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def emit_step_text(
        self, problem: Problem, state: Tensor, action: int
    ) -> str:
        """Render the next step's sentence from the IMAGINED state.

        ``predictor(state, u_action)`` is what the model believes the world
        looks like after doing ``action``; the frozen decoder writes that
        belief out as text.  The environment is never asked.
        """
        imagined = self._imagined_state(problem, state, [action])
        ids = self.decoder.generate(imagined, greedy=True)[0]
        return self.vocab.decode(ids)

    @torch.no_grad()
    def rollout_episode(
        self, problem: Problem, slack: int = 0, seed: int = 0
    ) -> AutonomousEpisode:
        prompt = prompt_sentences(problem, random.Random(seed))
        prompt_tokens = self._tokens(prompt)
        prompt_mask = torch.ones(
            1, len(prompt), dtype=torch.bool, device=self.device
        )
        budget = problem.n_necessary_steps + slack
        step_texts: list[str] = []
        actions: list[int] = []
        n_well_formed = n_action_match = n_value_correct = 0
        first_divergence: int | None = None
        claimed = stalled = False
        answer: int | None = None

        while len(step_texts) < budget:
            s = self._current_state(prompt_tokens, prompt_mask, step_texts)
            s0 = self._s0(prompt_tokens, prompt_mask)
            state_history, action_codes = self._causal_history(
                prompt_tokens, prompt_mask, step_texts, problem, actions
            )
            executed = frozenset(actions)
            root_candidates, _, _ = self.proposer.propose(
                s, problem, executed, top_k=self.prior_top_k,
                generator=self._proposal_generator(seed, len(step_texts)),
            )
            best = self._beam_search(
                s, s0, problem, executed, None, state_history, action_codes,
                score_seed=f"{seed}:{len(step_texts)}:scores",
                root_candidates=root_candidates,
            )
            chosen = best[0]
            if chosen is None:
                stalled = True
                break
            text = self.emit_step_text(problem, s, chosen)
            step_texts.append(text)
            actions.append(chosen)

            # ---- measurement only, from here to the end of the loop body --
            idx, value = parse_step_sentence(problem, text)
            well_formed = idx is not None and value is not None
            n_well_formed += int(well_formed)
            n_action_match += int(idx is not None and idx == chosen)
            correct = well_formed and value == problem.values[idx]
            n_value_correct += int(correct)
            if not correct and first_divergence is None:
                first_divergence = len(step_texts) - 1
            # The model's own completion claim: it wrote the step sentence of
            # the queried variable (the query is stated in the prompt, so this
            # reads nothing privileged).
            if idx is not None and idx == problem.query:
                claimed = True
                answer = value
                if self.stop_on_claim:
                    break
                # ``stop_on_claim=False`` keeps writing to the budget and
                # reports the LAST answer sentence, so an early, premature
                # claim can still be revised by later work.

        answer_head_correct: bool | None = None
        if getattr(self.decoder, "answer_head", None) is not None:
            final = self._current_state(prompt_tokens, prompt_mask, step_texts)
            head = int(self.decoder.answer_logits(final).argmax(-1).item())
            answer_head_correct = head == problem.answer % problem.modulus
        return AutonomousEpisode(
            answer_correct=(
                answer is not None and answer == problem.answer
            ),
            claimed_completion=claimed,
            answer=answer,
            true_answer=problem.answer,
            n_steps=len(step_texts),
            n_necessary=problem.n_necessary_steps,
            n_well_formed=n_well_formed,
            n_action_match=n_action_match,
            n_value_correct=n_value_correct,
            first_divergence=first_divergence,
            stalled=stalled,
            answer_head_correct=answer_head_correct,
            texts=step_texts,
            actions=actions,
        )


def aggregate_autonomous(
    episodes: list[AutonomousEpisode],
    *,
    slack: int = 0,
    slack_curve: bool = False,
) -> dict[str, object]:
    """Final-answer accuracy plus the diagnostics that localize the failure."""
    n = max(len(episodes), 1)
    steps = max(sum(e.n_steps for e in episodes), 1)
    heads = [
        e.answer_head_correct for e in episodes
        if e.answer_head_correct is not None
    ]
    divergences = [
        e.first_divergence for e in episodes if e.first_divergence is not None
    ]
    metrics: dict[str, object] = {
        # THE headline number: final answer vs ground truth, nothing else.
        "answer_accuracy": sum(e.answer_correct for e in episodes) / n,
        "completion_claim_rate": sum(e.claimed_completion for e in episodes) / n,
        "stall_rate": sum(e.stalled for e in episodes) / n,
        "mean_steps": sum(e.n_steps for e in episodes) / n,
        "mean_necessary": sum(e.n_necessary for e in episodes) / n,
        # Emitted-text diagnostics, per emitted step.
        "well_formed_rate": sum(e.n_well_formed for e in episodes) / steps,
        "action_match_rate": sum(e.n_action_match for e in episodes) / steps,
        "value_correct_rate": sum(e.n_value_correct for e in episodes) / steps,
        # Where the self-rollout first says something wrong (1-based step).
        "mean_first_divergence": (
            sum(divergences) / len(divergences) + 1.0 if divergences else None
        ),
        "never_diverged_rate": (n - len(divergences)) / n,
    }
    if heads:
        # Cheap upper-bound reference: the linear answer head read off the
        # final SELF-GENERATED state instead of the emitted text.
        metrics["answer_head_accuracy"] = sum(heads) / len(heads)
    if slack_curve:
        # Success here means "answered correctly within optimal + s steps".
        results = [
            EpisodeResult(
                e.answer_correct, e.n_steps, e.n_necessary, 0, 0,
                solved_at=e.n_steps if e.answer_correct else None,
            )
            for e in episodes
        ]
        metrics.update(slack_curve_metrics(results, slack))
    return metrics


def evaluate_autonomous(
    planner: AutonomousRollout,
    dataset,
    n_episodes: int,
    slack: int = 0,
    seed: int = 0,
    slack_curve: bool = False,
) -> dict[str, dict[str, object]]:
    episodes = [
        planner.rollout_episode(
            dataset.problem(i)[0], slack=slack, seed=seed + i
        )
        for i in range(n_episodes)
    ]
    return {
        "autonomous": aggregate_autonomous(
            episodes, slack=slack, slack_curve=slack_curve
        )
    }
