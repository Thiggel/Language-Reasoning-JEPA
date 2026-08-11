"""Open-ended action proposal from an eval-time learned action codebook."""

import pytest
import torch

from textjepa.data.igsm import env as igsm_env
from textjepa.data.igsm.dataset import IGSMDataset, build_vocab
from textjepa.data.igsm.render import action_phrase
from textjepa.models import DiscourseJEPA
from textjepa.planning import search as search_module
from textjepa.planning.codebook import CodebookCycleProposer, fit_codebook
from textjepa.planning.search import LatentPlanner


def _tiny_model(vocab):
    return DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=32, chunk_layers=1, chunk_heads=2,
        state_layers=1, state_heads=2, d_action=8, d_macro=4,
        predictor_kind="concat", observed_action_ldad=True,
        max_chunk_len=64,
    ).eval()


def _phrase_decoder(model, vocab, ids):
    """A decoder whose greedy output is always the phrase with token ``ids``."""
    def decoder(delta):
        logits = torch.zeros(
            len(delta), model.observed_action_decoder.max_len, len(vocab)
        )
        for position, token in enumerate(ids):
            logits[:, position, token] = 10.0
        logits[:, len(ids):, vocab.pad_id] = 10.0
        return logits

    return decoder


# ---------------------------------------------------------------- k-means
def _four_clusters(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    means = torch.tensor([
        [10.0, 0.0], [-10.0, 0.0], [0.0, 10.0], [0.0, -10.0],
    ])
    points = torch.cat([
        mean + 0.05 * torch.randn(40, 2, generator=generator)
        for mean in means
    ])
    return points, means


def test_kmeans_recovers_four_well_separated_clusters():
    points, means = _four_clusters()
    codes = fit_codebook(points, k=4, seed=0)
    assert codes.shape == (4, 2)
    # Each true mean has a centroid essentially on top of it, and no two
    # centroids collapse onto the same cluster.
    matched = torch.cdist(means, codes).argmin(-1)
    assert sorted(matched.tolist()) == [0, 1, 2, 3]
    assert torch.cdist(means, codes).min(-1).values.max() < 0.1


def test_kmeans_is_deterministic_given_the_seed():
    points, _ = _four_clusters()
    first = fit_codebook(points, k=4, seed=0)
    second = fit_codebook(points, k=4, seed=0)
    assert torch.equal(first, second)


def test_fit_codebook_clamps_k_to_the_available_embeddings():
    points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    assert fit_codebook(points, k=8, seed=0).shape == (2, 2)
    with pytest.raises(ValueError):
        fit_codebook(torch.zeros(0, 2), k=2, seed=0)


def test_proposer_requires_a_fitted_codebook_before_proposing():
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=17).problem(0)
    proposer = CodebookCycleProposer(
        _tiny_model(vocab), vocab, torch.device("cpu"), k=4
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
        CodebookCycleProposer(model, vocab, torch.device("cpu"))


# ------------------------------------------------------- no oracle access
def test_codebook_cycle_episode_never_touches_the_feasibility_oracle():
    """The proposal path may not read the feasible menu or reference graph.

    The only legitimate consumers of feasibility are the labeled diagnostics
    (proposal recall, invalid-step counting) in ``plan_episode``, never the
    proposer itself, and the menu builder ``_feasible`` must not run at all.
    """
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=31)
    problem, _ = dataset.problem(0)
    model = _tiny_model(vocab)
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"),
        candidate_interface="codebook_cycle", codebook_k=6, prior_top_k=4,
    )
    planner.fit_action_prior([dataset.problem(i)[0] for i in (1, 2, 3)])
    model.observed_action_decoder.forward = _phrase_decoder(
        model, vocab, vocab.encode(action_phrase(problem, problem.vars[0].idx))
    )

    inside_proposal = {"flag": False}
    oracle_calls_inside = []
    original_feasible = igsm_env.SymbolicEnv.feasible_actions
    original_propose = planner.proposer.propose

    def spy_feasible(self):
        if inside_proposal["flag"]:
            oracle_calls_inside.append(True)
        return original_feasible(self)

    def spy_propose(*args, **kwargs):
        inside_proposal["flag"] = True
        try:
            return original_propose(*args, **kwargs)
        finally:
            inside_proposal["flag"] = False

    def no_menu(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("codebook_cycle built the oracle feasible menu")

    igsm_env.SymbolicEnv.feasible_actions = spy_feasible
    planner.proposer.propose = spy_propose
    original_module_feasible = search_module._feasible
    search_module._feasible = no_menu
    try:
        result = planner.plan_episode(problem, slack=1, seed=0)
    finally:
        igsm_env.SymbolicEnv.feasible_actions = original_feasible
        search_module._feasible = original_module_feasible

    assert not oracle_calls_inside
    assert result.proposal_recall is not None
    assert result.parse_rate is not None
    counts = result.proposal_counts
    assert counts is not None
    assert counts["n_proposed"] == 6.0  # one score per code
    assert counts["n_parseable"] + counts["n_unparseable"] == 6.0


# ------------------------------------------------------- executed masking
def test_codebook_cycle_masks_executed_actions_at_roots_and_expansions():
    """Proposals never re-propose an action the planner already took.

    Same mask as ``ldad_cycle``/``cem_cycle``: the planner's resolved set at
    beam roots, plus the beam-imagined actions at expansions.
    """
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=37)
    problem, _ = dataset.problem(0)
    model = _tiny_model(vocab)
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=2,
        candidate_interface="codebook_cycle", codebook_k=4,
    )
    planner.fit_action_prior([dataset.problem(i)[0] for i in (1, 2, 3)])
    action = problem.vars[0].idx
    model.observed_action_decoder.forward = _phrase_decoder(
        model, vocab, vocab.encode(action_phrase(problem, action))
    )
    roots = planner._proposer_candidates(problem, torch.zeros(32), frozenset())
    assert roots == [action]
    masked = planner._proposer_candidates(
        problem, torch.zeros(32), frozenset({action})
    )
    assert masked == []


# ------------------------------------------------------- reproducibility
def test_codebook_cycle_proposals_are_reproducible():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=41)
    problem, _ = dataset.problem(0)
    train = [dataset.problem(i)[0] for i in (1, 2, 3)]

    def run():
        torch.manual_seed(0)
        model = _tiny_model(vocab)
        torch.manual_seed(12345)  # ambient RNG must not matter
        proposer = CodebookCycleProposer(
            model, vocab, torch.device("cpu"), k=6, seed=3
        )
        proposer.fit_prior(train)
        state = torch.zeros(32)
        return proposer.codebook, proposer.propose(state, problem, frozenset())

    codes_a, (actions_a, parsed_a, diag_a) = run()
    codes_b, (actions_b, parsed_b, diag_b) = run()
    assert torch.equal(codes_a, codes_b)
    assert actions_a == actions_b
    assert parsed_a == parsed_b
    assert diag_a == diag_b
