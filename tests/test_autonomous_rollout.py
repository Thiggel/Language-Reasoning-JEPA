"""Menu-free, oracle-free autonomous self-rollout (candidate_interface=autonomous)."""

import pytest
import torch

from textjepa.data.igsm import env as igsm_env
from textjepa.data.igsm.dataset import IGSMDataset, build_vocab
from textjepa.data.igsm.render import action_phrase, step_sentence
from textjepa.models import DiscourseJEPA
from textjepa.models.state_decoder import FrozenStateSentenceDecoder
from textjepa.planning.autonomous import (
    AutonomousRollout,
    aggregate_autonomous,
    evaluate_autonomous,
    parse_step_sentence,
)


def _tiny_model(vocab):
    return DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=32, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        predictor_kind="concat", observed_action_ldad=True,
        max_chunk_len=64,
    ).eval()


def _tiny_decoder(vocab):
    return FrozenStateSentenceDecoder(
        d_state=32, vocab_size=len(vocab), max_len=8, n_layers=1, n_heads=2,
        n_answers=23,
    ).eval()


def _planner(vocab, dataset, decoder=None, **kwargs):
    model = _tiny_model(vocab)
    planner = AutonomousRollout(
        model, vocab, torch.device("cpu"),
        decoder if decoder is not None else _tiny_decoder(vocab),
        codebook_k=6, prior_top_k=4, **kwargs,
    )
    planner.fit_action_prior([dataset.problem(i)[0] for i in (1, 2, 3)])
    return planner


class _ScriptedRollout(AutonomousRollout):
    """Rollout whose renderer is replaced by a scripted text source.

    Isolates the loop (proposal -> planning -> self-emitted text -> re-encode)
    from the decoder's quality, so the control flow and the diagnostics can be
    tested against a known-good and a known-bad renderer.
    """

    def __init__(self, *args, render=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.render = render

    def emit_step_text(self, problem, state, action):
        return self.render(problem, action)


# ------------------------------------------------------------ step parsing
def test_parse_step_sentence_reads_variable_and_value():
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=17).problem(0)
    for var in problem.vars:
        idx, value = parse_step_sentence(problem, step_sentence(problem, var.idx))
        assert idx == var.idx
        assert value == problem.values[var.idx]


def test_parse_step_sentence_rejects_ill_formed_text():
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=17).problem(0)
    name = problem.vars[0].name
    assert parse_step_sentence(problem, "") == (None, None)
    assert parse_step_sentence(problem, action_phrase(problem, 0)) == (None, None)
    # Known variable, no stated number.
    assert parse_step_sentence(problem, f"so the number of {name} is .") == (0, None)
    # Well-formed shape, variable not in this problem.
    assert parse_step_sentence(
        problem, "so the number of purple nothings is 3 ."
    ) == (None, 3)


# ------------------------------------------------------------- the loop
def test_perfect_renderer_solves_and_claims_completion():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=31)
    problem, _ = dataset.problem(0)
    planner = _ScriptedRollout(
        _tiny_model(vocab), vocab, torch.device("cpu"), _tiny_decoder(vocab),
        codebook_k=6, prior_top_k=4, render=step_sentence,
    )
    planner.fit_action_prior([dataset.problem(i)[0] for i in (1, 2, 3)])
    episode = planner.rollout_episode(problem, slack=len(problem.vars), seed=0)

    assert episode.n_steps == len(episode.texts) == len(episode.actions)
    # Every emitted sentence is well formed, is about the action that was
    # chosen, and states the right number.
    assert episode.n_well_formed == episode.n_steps
    assert episode.n_action_match == episode.n_steps
    assert episode.n_value_correct == episode.n_steps
    assert episode.first_divergence is None
    if episode.claimed_completion:
        assert episode.answer == problem.answer
        assert episode.answer_correct
        assert episode.actions[-1] == problem.query
    else:
        assert episode.answer is None and not episode.answer_correct


def test_broken_renderer_is_localized_by_the_diagnostics():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=31)
    problem, _ = dataset.problem(0)
    planner = _ScriptedRollout(
        _tiny_model(vocab), vocab, torch.device("cpu"), _tiny_decoder(vocab),
        codebook_k=6, prior_top_k=4,
        render=lambda p, a: "so the number of unknown things is .",
    )
    planner.fit_action_prior([dataset.problem(i)[0] for i in (1, 2, 3)])
    episode = planner.rollout_episode(problem, slack=1, seed=0)

    # Never claims completion: it runs to the budget, or stalls when the
    # codebook has no unexecuted proposal left. It never falls back to a menu.
    assert episode.n_steps <= problem.n_necessary_steps + 1
    assert episode.stalled or episode.n_steps == problem.n_necessary_steps + 1
    assert episode.n_well_formed == 0
    assert episode.n_value_correct == 0
    assert episode.first_divergence == 0
    assert not episode.claimed_completion
    assert episode.answer is None and not episode.answer_correct


def test_value_error_renderer_diverges_at_the_first_wrong_number():
    """Right sentence shape, wrong arithmetic: exactly the predicted failure."""
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=31)
    problem, _ = dataset.problem(0)

    def wrong_after_first(p, a):
        text = step_sentence(p, a)
        state = wrong_after_first
        state.n = getattr(state, "n", 0) + 1
        if state.n <= 1:
            return text
        words = text.split()
        words[-2] = str((int(words[-2]) + 1) % p.modulus)
        return " ".join(words)

    planner = _ScriptedRollout(
        _tiny_model(vocab), vocab, torch.device("cpu"), _tiny_decoder(vocab),
        codebook_k=6, prior_top_k=4, render=wrong_after_first,
    )
    planner.fit_action_prior([dataset.problem(i)[0] for i in (1, 2, 3)])
    episode = planner.rollout_episode(problem, slack=2, seed=0)
    assert episode.n_well_formed == episode.n_steps
    assert episode.first_divergence == 1
    assert episode.n_value_correct == 1


def test_stop_on_claim_false_runs_to_the_budget():
    """Without early stopping the rollout keeps writing and reports the last claim."""
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=31)
    problem, _ = dataset.problem(0)
    planner = _ScriptedRollout(
        _tiny_model(vocab), vocab, torch.device("cpu"), _tiny_decoder(vocab),
        codebook_k=256, prior_top_k=8, stop_on_claim=False,
        render=step_sentence,
    )
    planner.fit_action_prior([dataset.problem(i)[0] for i in (1, 2, 3)])
    budget = problem.n_necessary_steps
    episode = planner.rollout_episode(problem, slack=0, seed=0)
    assert episode.stalled or episode.n_steps == budget
    if episode.claimed_completion:
        # The answer comes from the LAST query sentence written, not the first.
        assert episode.answer == problem.answer


# ---------------------------------------------------- no oracle executor
def test_rollout_never_calls_the_oracle_executor():
    """The environment must not step, render an outcome, or reveal feasibility."""
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=37)
    problem, _ = dataset.problem(0)
    planner = _planner(vocab, dataset)

    def forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the autonomous rollout used the oracle executor")

    originals = (
        igsm_env.SymbolicEnv.step,
        igsm_env.SymbolicEnv.step_or_invalid,
        igsm_env.SymbolicEnv.feasible_actions,
    )
    igsm_env.SymbolicEnv.step = forbidden
    igsm_env.SymbolicEnv.step_or_invalid = forbidden
    igsm_env.SymbolicEnv.feasible_actions = forbidden
    try:
        episode = planner.rollout_episode(problem, slack=1, seed=0)
    finally:
        (
            igsm_env.SymbolicEnv.step,
            igsm_env.SymbolicEnv.step_or_invalid,
            igsm_env.SymbolicEnv.feasible_actions,
        ) = originals
    assert episode.n_steps >= 1
    assert all(isinstance(text, str) for text in episode.texts)


# ------------------------------------------------------- frozen / JEPA-pure
def test_backbone_and_decoder_are_frozen():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=37)
    model = _tiny_model(vocab)
    for parameter in model.parameters():  # a trainable backbone must be frozen
        parameter.requires_grad_(True)
    decoder = _tiny_decoder(vocab)
    planner = AutonomousRollout(
        model, vocab, torch.device("cpu"), decoder, codebook_k=4
    )
    assert not any(p.requires_grad for p in planner.model.parameters())
    assert not any(p.requires_grad for p in planner.decoder.parameters())
    assert not planner.model.training and not planner.decoder.training
    planner.fit_action_prior([dataset.problem(i)[0] for i in (1, 2, 3)])
    episode = planner.rollout_episode(dataset.problem(0)[0], slack=0, seed=0)
    assert episode.n_steps >= 0


def test_oracle_energies_are_rejected():
    vocab = build_vocab(23)
    with pytest.raises(ValueError):
        AutonomousRollout(
            _tiny_model(vocab), vocab, torch.device("cpu"),
            _tiny_decoder(vocab), energy="oracle_goal",
        )
    with pytest.raises(ValueError):
        AutonomousRollout(
            _tiny_model(vocab), vocab, torch.device("cpu"),
            _tiny_decoder(vocab), candidate_interface="feasible_menu",
        )


# ------------------------------------------------------------ determinism
def test_rollout_is_deterministic_given_the_seed():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=41)
    problem, _ = dataset.problem(0)

    def run():
        torch.manual_seed(0)
        model = _tiny_model(vocab)
        torch.manual_seed(1)
        decoder = _tiny_decoder(vocab)
        torch.manual_seed(12345)  # ambient RNG must not matter
        planner = AutonomousRollout(
            model, vocab, torch.device("cpu"), decoder, codebook_k=6,
        )
        planner.fit_action_prior([dataset.problem(i)[0] for i in (1, 2, 3)])
        return planner.rollout_episode(problem, slack=1, seed=7)

    first, second = run(), run()
    assert first.texts == second.texts
    assert first.actions == second.actions
    assert first.answer == second.answer


# ------------------------------------------------------------- aggregation
def test_evaluate_autonomous_reports_answer_accuracy_and_diagnostics():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=6, seed=43)
    planner = _planner(vocab, dataset)
    results = evaluate_autonomous(
        planner, dataset, n_episodes=2, slack=1, seed=0, slack_curve=True
    )
    metrics = results["autonomous"]
    for key in (
        "answer_accuracy", "completion_claim_rate", "well_formed_rate",
        "action_match_rate", "value_correct_rate", "never_diverged_rate",
        "answer_head_accuracy", "success_by_slack",
    ):
        assert key in metrics
    assert 0.0 <= metrics["answer_accuracy"] <= 1.0


def test_aggregate_handles_an_empty_run():
    metrics = aggregate_autonomous([])
    assert metrics["answer_accuracy"] == 0.0
    assert metrics["mean_first_divergence"] is None
