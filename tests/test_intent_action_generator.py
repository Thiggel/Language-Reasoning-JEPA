"""State-conditioned intent-phrase generator: head, detach discipline, parse.

The generator head answers "which intent phrase comes next in this state?"
with free text, so a planner can propose actions with no candidate catalogue
at all (``candidate_interface=generator_cycle``: sample phrases, parse them
against the problem's action space, rank survivors by LDAD
cycle-consistency).  It is an auxiliary read-out: the state it conditions on
is detached, so it can never shape the JEPA representation.
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab, collate
from textjepa.data.igsm.render import action_phrase, parse_action_phrase
from textjepa.models import DiscourseJEPA
from textjepa.objectives import ActionGeneration
from textjepa.planning import LatentPlanner


def _tiny_model(vocab, **kwargs) -> DiscourseJEPA:
    return DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=64,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        predictor_layers=1, predictor_heads=2, d_action=8, d_macro=4,
        **kwargs,
    )


# --------------------------------------------------------------------- #
# head + training signal
# --------------------------------------------------------------------- #
def test_generator_head_is_absent_by_default_and_objective_is_inert():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=2, seed=3)
    batch = collate([dataset[0], dataset[1]], vocab.pad_id)
    model = _tiny_model(vocab)
    assert model.action_generator is None
    out = model(batch)
    assert "action_generator_logits" not in out.extras
    assert float(ActionGeneration()(out, batch)) == 0.0
    with pytest.raises(RuntimeError):
        model.generate_action_phrases(out.prev_states[0, 0])


def test_generator_forward_backward_is_finite_and_shaped():
    torch.manual_seed(0)
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=2, seed=3)
    batch = collate([dataset[0], dataset[1]], vocab.pad_id)
    model = _tiny_model(vocab, action_generator=True)
    out = model(batch)
    logits = out.extras["action_generator_logits"]
    B, T, L = batch["action_tokens"].shape
    assert logits.shape == (B, T, L, len(vocab))
    assert out.extras["action_generator_valid"].shape == (B, T)
    loss = ActionGeneration()(out, batch)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.action_generator.parameters()
        if p.requires_grad
    )


def test_generator_loss_sends_no_gradient_into_encoder_or_predictor():
    torch.manual_seed(0)
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=2, seed=3)
    batch = collate([dataset[0], dataset[1]], vocab.pad_id)
    model = _tiny_model(vocab, action_generator=True)
    out = model(batch)
    ActionGeneration()(out, batch).backward()
    representation = (
        list(model.chunk_encoder.named_parameters())
        + list(model.state_model.named_parameters())
        + list(model.core.predictor.named_parameters())
        + list(model.action_encoder.named_parameters())
    )
    for name, parameter in representation:
        assert parameter.grad is None or parameter.grad.abs().max() == 0.0, (
            f"generator loss leaked gradient into {name}"
        )


def test_generator_loss_decreases_on_a_repeated_batch():
    torch.manual_seed(0)
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=2, seed=3)
    batch = collate([dataset[0], dataset[1]], vocab.pad_id)
    model = _tiny_model(vocab, action_generator=True)
    objective = ActionGeneration()
    optimizer = torch.optim.Adam(model.action_generator.parameters(), lr=3e-3)
    first = None
    for _ in range(30):
        loss = objective(model(batch), batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first = loss.item() if first is None else first
    assert loss.item() < first


# --------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------- #
def test_generate_action_phrases_returns_deduplicated_token_sequences():
    torch.manual_seed(0)
    vocab = build_vocab(23)
    model = _tiny_model(vocab, action_generator=True).eval()
    state = torch.randn(1, 64)
    phrases = model.generate_action_phrases(
        state, k=8, top_p=0.9, max_len=12, temperature=1.0
    )
    assert 1 <= len(phrases) <= 8
    assert len({tuple(p) for p in phrases}) == len(phrases)
    for ids in phrases:
        assert ids and len(ids) <= 12
        assert all(0 < i < len(vocab) for i in ids)
        assert isinstance(vocab.decode(ids), str)


def test_generator_sampling_rejects_invalid_arguments():
    vocab = build_vocab(23)
    model = _tiny_model(vocab, action_generator=True).eval()
    state = torch.randn(64)
    for kwargs in ({"k": 0}, {"top_p": 0.0}, {"temperature": 0.0}):
        with pytest.raises(ValueError):
            model.generate_action_phrases(state, **kwargs)


# --------------------------------------------------------------------- #
# parse round trip
# --------------------------------------------------------------------- #
def test_action_phrase_parse_round_trip_over_the_whole_catalogue():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=4, seed=11)
    for index in range(4):
        problem, _ = dataset.problem(index)
        for action in range(len(problem.vars)):
            phrase = action_phrase(problem, action)
            assert parse_action_phrase(problem, phrase) == action
            # tokenizer round trip (the planner parses decoded samples)
            assert parse_action_phrase(
                problem, vocab.decode(vocab.encode(phrase))
            ) == action


def test_parse_rejects_ill_formed_or_contradictory_phrases():
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=11).problem(0)
    internal = next(v for v in problem.vars if not v.is_leaf)
    leaf = next(v for v in problem.vars if v.is_leaf)
    a, b = (problem.vars[p] for p in internal.parents)
    for text in (
        "",
        "derive .",
        "look up the number of ghost apples .",
        # a real action's words in an impossible combination
        f"derive {internal.name} from {a.name} plus {b.name} plus {b.name} .",
        # leaf variables have no derivation
        f"derive {leaf.name} from {a.name} plus {b.name} .",
        # right variable name, wrong parents order
        f"derive {internal.name} from {b.name} plus {a.name} .",
    ):
        parsed = parse_action_phrase(problem, text)
        assert parsed is None or parsed != internal.idx or (
            # swapped parents are legitimate only for a symmetric definition
            internal.parents == (internal.parents[1], internal.parents[0])
        )
    assert parse_action_phrase(problem, f"look up the number of {a.name}") in (
        None, a.idx
    )


# --------------------------------------------------------------------- #
# open-ended planning
# --------------------------------------------------------------------- #
def test_generator_cycle_planning_requires_both_auxiliary_heads():
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=3).problem(0)
    model = _tiny_model(vocab, action_generator=True).eval()  # no LDAD decoder
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=1,
        candidate_interface="generator_cycle", prior_top_k=2,
    )
    with pytest.raises(RuntimeError):
        planner.plan_episode(problem, slack=1, seed=0)


def _stub_proposals(planner, problem, vocab, garbage: bool = True):
    """Make the generator emit this problem's phrases plus one non-phrase.

    An untrained head almost never writes a parseable phrase, so the planning
    interface (parse, discard, cycle-consistency ranking, diagnostics) is
    exercised with a stubbed sampler; the real head's sampling is covered
    above.
    """
    phrases = [
        vocab.encode(action_phrase(problem, action))
        for action in range(len(problem.vars))
    ]
    if garbage:
        phrases.append(vocab.encode("derive nothing at all ."))
    planner.model.generate_action_phrases = (
        lambda state, k=8, top_p=1.0, max_len=None, temperature=1.0,
        generator=None: phrases
    )


def test_generator_cycle_planning_runs_and_reports_proposal_recall():
    torch.manual_seed(0)
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=3).problem(0)
    model = _tiny_model(
        vocab, action_generator=True, observed_action_ldad=True
    ).eval()
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=1,
        candidate_interface="generator_cycle", prior_top_k=2,
        generator_samples=8, generator_top_p=0.95,
    )
    _stub_proposals(planner, problem, vocab)
    with torch.no_grad():
        result = planner.plan_episode(problem, slack=1, seed=0)
    assert result.steps >= 1
    # every feasible action is among the proposals, so recall is 1; exactly
    # one of the n_vars + 1 samples is unparseable.
    assert result.proposal_recall == 1.0
    assert result.parse_rate == pytest.approx(
        len(problem.vars) / (len(problem.vars) + 1)
    )


def test_generator_cycle_stalls_gracefully_when_nothing_parses():
    torch.manual_seed(0)
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=3).problem(0)
    model = _tiny_model(
        vocab, action_generator=True, observed_action_ldad=True
    ).eval()
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=1,
        candidate_interface="generator_cycle", prior_top_k=2,
        generator_samples=4,
    )
    planner.model.generate_action_phrases = (
        lambda *a, **kw: [vocab.encode("derive nothing at all .")]
    )
    with torch.no_grad():
        result = planner.plan_episode(problem, slack=1, seed=0)
    assert not result.solved and result.steps == 0
    assert result.proposal_recall == 0.0 and result.parse_rate == 0.0


def test_generator_cycle_beam_depth_two_expands_from_imagined_states():
    torch.manual_seed(0)
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=5).problem(0)
    model = _tiny_model(
        vocab, action_generator=True, observed_action_ldad=True
    ).eval()
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=2, max_expand=4,
        candidate_interface="generator_cycle", prior_top_k=2,
        generator_samples=6, search_algorithm="beam",
    )
    _stub_proposals(planner, problem, vocab)
    with torch.no_grad():
        result = planner.plan_episode(problem, slack=1, seed=0)
    assert result.steps >= 1


def test_generator_cycle_metrics_carry_proposal_recall():
    torch.manual_seed(0)
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=2, seed=3)
    model = _tiny_model(
        vocab, action_generator=True, observed_action_ldad=True
    ).eval()
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=1,
        candidate_interface="generator_cycle", prior_top_k=2,
        generator_samples=6,
    )
    from textjepa.planning import evaluate_planning

    with torch.no_grad():
        results = evaluate_planning(planner, dataset, 2, slack=1, seed=0)
    metrics = next(iter(results.values()))
    assert "proposal_recall" in metrics
    assert "proposal_parse_rate" in metrics


# --------------------------------------------------------------------- #
# config / launcher plumbing
# --------------------------------------------------------------------- #
def test_configs_expose_the_generator_head_and_planner_knobs():
    root = Path(__file__).resolve().parents[1]
    model = OmegaConf.load(root / "configs/model/discourse.yaml")
    assert model.action_generator is False  # off by default
    objective = OmegaConf.load(root / "configs/objective/discourse.yaml")
    assert objective.action_generator.weight == 0.25
    assert (
        objective.action_generator._target_
        == "textjepa.objectives.ActionGeneration"
    )
    plan = OmegaConf.load(root / "configs/plan.yaml")
    assert plan.generator_samples == 16
    assert plan.generator_top_p == 1.0
    assert plan.generator_temperature == 1.0


def test_horizon_energy_cell_wires_the_ldad_generator_variant():
    root = Path(__file__).resolve().parents[1]
    cell = (root / "scripts/run_intent_horizon_energy_cell.sh").read_text()
    assert "mix4_aux025_nohorizon_ldad_gen)" in cell
    assert "model.observed_action_ldad=true" in cell
    assert "objective.observed_action_ldad.weight=1.0" in cell
    assert "model.action_generator=true" in cell
    assert "eval_candidate_interface=generator_cycle" in cell
    assert 'GENERATOR_SAMPLES="${GENERATOR_SAMPLES:-16}"' in cell
    evaluation = (
        root / "scripts/run_intent_terminal_energy_eval.sh"
    ).read_text()
    assert "generator_samples=${GENERATOR_SAMPLES:-16}" in evaluation
    assert 'generator_samples="$generator_samples"' in evaluation
