from dataclasses import asdict
import json

import pytest

from scripts.validate_intent_reasoning_data import validate
from textjepa.data.planbench import (
    compile_blocksworld_episode,
    parse_blocksworld_pddl,
)


PDDL = """
(define (problem tiny)
  (:domain blocksworld-4ops)
  (:objects a b)
  (:init (ontable a) (ontable b) (clear a) (clear b) (handempty))
  (:goal (and (on a b))))
"""


def _write(path, split):
    problem = parse_blocksworld_pddl(PDDL, problem_id=f"tiny-{split}")
    episode = compile_blocksworld_episode(problem, split, counterfactual_k=1)
    path.write_text(json.dumps(asdict(episode)) + "\n")


def test_generic_reasoning_validator_replays_and_separates_splits(tmp_path):
    datasets = []
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.jsonl"
        _write(path, split)
        datasets.append((split, path))
    result = validate(datasets, "planbench-blocksworld")
    assert result["split_identity_disjoint"]
    assert all(
        item["expert_catalogue_recall"] == 1.0
        and item["goal_success_rate"] == 1.0
        for item in result["splits"].values()
    )


def test_generic_reasoning_validator_rejects_cross_split_identity(tmp_path):
    problem = parse_blocksworld_pddl(PDDL, problem_id="same")
    paths = []
    for split in ("train", "val"):
        path = tmp_path / f"{split}.jsonl"
        episode = compile_blocksworld_episode(
            problem, split, counterfactual_k=0
        )
        path.write_text(json.dumps(asdict(episode)) + "\n")
        paths.append((split, path))
    with pytest.raises(RuntimeError, match="overlaps"):
        validate(paths, "planbench-blocksworld")
