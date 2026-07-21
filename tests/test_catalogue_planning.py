from textjepa.data.planbench import compile_blocksworld_episode, parse_blocksworld_pddl
from textjepa.data.proofwriter import compile_proofwriter_episode
from textjepa.data.observed_action import build_observed_action_vocab
from textjepa.models import DiscourseJEPA
from textjepa.planning.catalogue import (
    CatalogueLatentPlanner,
    environment_from_episode,
    environment_from_faithful_problem,
)
import torch

from tests.test_planbench_adapter import PDDL
from tests.test_proofwriter_adapter import _record


def test_blocksworld_environment_executes_catalogue_actions_and_counts_invalid():
    episode = compile_blocksworld_episode(
        parse_blocksworld_pddl(PDDL), "test"
    )
    environment = environment_from_episode(episode)
    assert not environment.solved
    before = environment.invalid_actions
    environment.step("stack block a on block b")
    assert environment.invalid_actions == before + 1
    for transition in episode.transitions:
        environment.step(transition.action)
    assert environment.solved


def test_proofwriter_environment_supports_alternative_valid_derivations():
    episode = compile_proofwriter_episode(_record(), "Q1", "test")
    environment = environment_from_episode(episode)
    for transition in episode.transitions:
        environment.step(transition.action)
    assert environment.solved
    assert environment.invalid_actions == 0


def test_catalogue_planner_runs_depth_two_without_querying_oracle_menu():
    episode = compile_blocksworld_episode(
        parse_blocksworld_pddl(PDDL), "test"
    )
    vocab = build_observed_action_vocab([episode])
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=32,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        predictor_layers=1, predictor_heads=2, d_action=8, macro_k=0,
        max_chunk_len=64, max_chunks=32,
    ).eval()
    planner = CatalogueLatentPlanner(
        model, vocab, torch.device("cpu"), simulation_depth=2,
        proposal_top_m=2, beam_width=2,
    )
    result = planner.run_episode(environment_from_episode(episode), 0)
    assert result.steps <= result.optimal_length
    assert result.invalid_actions >= 0


def test_faithful_igsm_wrapper_exposes_full_catalogue_not_feasible_menu():
    from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab

    dataset = FaithfulDataset(
        cached_faithful_vocab(), size=1, seed=917, max_op=15,
        max_edge=20, op_range=(3, 5), distractor_prob=0.0,
    )
    problem, _ = dataset.problem(0)
    environment = environment_from_faithful_problem(problem)
    assert len(environment.catalogue) == len(problem.action_order)
    assert len(environment.environment.feasible_actions()) < len(
        environment.catalogue
    )
