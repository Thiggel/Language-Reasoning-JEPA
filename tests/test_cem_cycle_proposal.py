"""Open-ended action proposal by CEM in the action-embedding space."""

import pytest
import torch

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab
from textjepa.data.igsm.graph import OP_WORDS
from textjepa.data.igsm.render import (
    action_phrase, catalogue_phrases, parse_action_phrase,
)
from textjepa.models import DiscourseJEPA
from textjepa.planning.cem_cycle import (
    ActionPrior,
    CEMCycleProposer,
    cem_gaussian,
    diagonal_prior,
)
from textjepa.planning.search import LatentPlanner


def _tiny_model(vocab):
    return DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=32, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        predictor_kind="concat", observed_action_ldad=True,
        max_chunk_len=64,
    ).eval()


def test_cem_converges_toward_a_quadratic_optimum():
    target = torch.tensor([2.0, -1.0, 0.5])
    prior = ActionPrior(torch.zeros(3), torch.ones(3) * 3.0)
    generator = torch.Generator().manual_seed(0)
    elites, scores = cem_gaussian(
        lambda batch: -(batch - target).square().sum(-1),
        prior, population=128, elites=16, iterations=8,
        prior_anchor=0.0, generator=generator,
    )
    start = -(prior.sample(128, torch.Generator().manual_seed(1)) - target)
    start_best = start.square().sum(-1).min()
    assert scores[0] >= scores[-1]
    assert (elites[0] - target).norm() < 0.5
    assert -scores[0] < start_best


def test_prior_anchor_keeps_the_population_near_the_training_gaussian():
    prior = ActionPrior(torch.zeros(2), torch.ones(2))
    far = torch.tensor([50.0, 50.0])
    generator = torch.Generator().manual_seed(0)
    anchored, _ = cem_gaussian(
        lambda batch: -(batch - far).square().sum(-1),
        prior, population=64, elites=8, iterations=5,
        prior_anchor=1.0, generator=generator,
    )
    assert anchored.mean(0).norm() < 5.0


def test_parse_round_trip_over_a_full_catalogue():
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=11).problem(0)
    phrases = catalogue_phrases(problem)
    assert len(phrases) == len(problem.vars)
    for idx, phrase in enumerate(phrases):
        assert phrase == action_phrase(problem, idx)
        assert parse_action_phrase(problem, phrase) == idx


def test_parse_rejects_phrases_outside_the_problem_action_space():
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=13).problem(0)
    internal = next(v for v in problem.vars if not v.is_leaf)
    a, b = (problem.vars[p].name for p in internal.parents)
    internal_op = OP_WORDS[internal.op]
    assert parse_action_phrase(problem, "") is None
    assert parse_action_phrase(problem, "derive nothing at all .") is None
    assert parse_action_phrase(
        problem, f"look up the number of {internal.name} ."
    ) is None
    wrong_op = "minus" if internal.op != "sub" else "plus"
    assert parse_action_phrase(
        problem, f"derive {internal.name} from {a} {wrong_op} {b} ."
    ) is None
    if a != b:  # parent order is part of the phrase, so a swap does not parse
        assert parse_action_phrase(
            problem, f"derive {internal.name} from {b} {internal_op} {a} ."
        ) is None


def test_proposer_requires_a_training_prior_before_proposing():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=2, seed=17)
    problem, _ = dataset.problem(0)
    proposer = CEMCycleProposer(
        _tiny_model(vocab), vocab, torch.device("cpu"), population=4,
        elites=2, iterations=1,
    )
    with pytest.raises(RuntimeError):
        proposer.propose(torch.zeros(32), problem, frozenset())


def test_proposer_rejects_checkpoints_without_an_ldad_decoder():
    vocab = build_vocab(23)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=32,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        d_action=8, d_macro=4, predictor_kind="concat",
    ).eval()
    with pytest.raises(RuntimeError):
        CEMCycleProposer(model, vocab, torch.device("cpu"))


def test_cem_cycle_proposes_parsed_candidates_and_reports_diagnostics():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=19)
    problem, _ = dataset.problem(0)
    model = _tiny_model(vocab)
    proposer = CEMCycleProposer(
        model, vocab, torch.device("cpu"), population=8, elites=3,
        iterations=2,
    )
    proposer.fit_prior([dataset.problem(i)[0] for i in (1, 2, 3)])

    # A random-weight decoder rarely emits a well-formed phrase, so drive the
    # cycle route with a decoder whose greedy output is a real phrase; the
    # search, grounding, and bookkeeping are what this test covers.
    phrase = action_phrase(problem, problem.vars[0].idx)
    ids = vocab.encode(phrase)

    def decoder(delta):
        logits = torch.zeros(
            len(delta), model.observed_action_decoder.max_len, len(vocab)
        )
        for position, token in enumerate(ids):
            logits[:, position, token] = 10.0
        logits[:, len(ids):, vocab.pad_id] = 10.0
        return logits

    model.observed_action_decoder.forward = decoder
    actions, parsed, diagnostics = proposer.propose(
        torch.zeros(32), problem, frozenset(),
        generator=torch.Generator().manual_seed(0),
    )
    assert actions == [problem.vars[0].idx]
    assert parsed == [problem.vars[0].idx]
    assert diagnostics["n_parseable"] == diagnostics["n_proposed"]
    assert diagnostics["n_unique"] == 1.0
    assert diagnostics["parse_rate"] == 1.0


def test_cem_cycle_episode_runs_end_to_end_and_records_diagnostics():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=23)
    problem, _ = dataset.problem(0)
    model = _tiny_model(vocab)
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), candidate_interface="cem_cycle",
        cem_population=6, cem_elites=2, cem_iters=1, prior_top_k=4,
    )
    planner.fit_action_prior([dataset.problem(i)[0] for i in (1, 2, 3)])
    result = planner.plan_episode(problem, slack=1, seed=0)
    counts = result.proposal_counts
    assert counts is not None
    assert counts["n_proposed"] == 2.0  # the elite population
    assert counts["n_parseable"] + counts["n_unparseable"] == 2.0
    assert result.proposal_recall is not None
    assert result.parse_rate is not None
    assert result.steps >= 0


def test_cem_cycle_masks_executed_actions_at_roots_and_expansions():
    """Proposals never re-propose an action the planner already took.

    The same executed-action mask ldad_cycle applies is applied here, at beam
    roots (the planner's resolved set) and at beam expansions (resolved plus
    the actions imagined earlier in the partial sequence).
    """
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=29)
    problem, _ = dataset.problem(0)
    model = _tiny_model(vocab)
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=2,
        candidate_interface="cem_cycle", cem_population=4, cem_elites=2,
        cem_iters=1,
    )
    planner.fit_action_prior([dataset.problem(i)[0] for i in (1, 2, 3)])

    # Decode every displacement to the same phrase: without masking it would
    # be proposed at the root and again at the expansion.
    ids = vocab.encode(action_phrase(problem, problem.vars[0].idx))

    def decoder(delta):
        logits = torch.zeros(
            len(delta), model.observed_action_decoder.max_len, len(vocab)
        )
        for position, token in enumerate(ids):
            logits[:, position, token] = 10.0
        logits[:, len(ids):, vocab.pad_id] = 10.0
        return logits

    model.observed_action_decoder.forward = decoder
    action = problem.vars[0].idx
    roots = planner._cem_candidates(problem, torch.zeros(32), frozenset())
    assert roots == [action]
    masked = planner._cem_candidates(
        problem, torch.zeros(32), frozenset({action})
    )
    assert masked == []


def test_cem_cycle_offmanifold_penalty_is_scale_invariant():
    """Rescaling the action embeddings must not rescale the penalty.

    The off-manifold distance is measured in units of the prior's mean
    embedding norm, so cem_offmanifold_lambda means the same thing for
    checkpoints whose action codes live at different magnitudes.
    """
    from textjepa.planning.cem_cycle import diagonal_prior

    codes = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    prior = diagonal_prior(codes)
    scaled = diagonal_prior(codes * 7.0)
    u, v = codes[0], codes[1]
    penalty = (u - v).square().sum() / prior.scale ** 2
    scaled_penalty = ((u - v) * 7.0).square().sum() / scaled.scale ** 2
    assert torch.allclose(penalty, scaled_penalty)


def test_diagonal_prior_matches_the_encoded_training_phrases():
    codes = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    prior = diagonal_prior(codes)
    assert torch.allclose(prior.mean, torch.tensor([1.0, 2.0]))
    assert torch.allclose(prior.std, codes.std(0, unbiased=False))
