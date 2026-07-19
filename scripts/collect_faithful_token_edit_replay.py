"""Collect target-free frozen-GAR trajectories for off-policy state replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf

try:
    from scripts.plan_faithful_token_edits import (
        buffer_key, copy_buffer, gar_scores,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from plan_faithful_token_edits import buffer_key, copy_buffer, gar_scores
from textjepa.data.faithful_token_edits import (
    _apply,
    _proposal_tokens,
    propose_deployable_edits,
)
from textjepa.data.token_edit_replay import REPLAY_FORMAT
from textjepa.utils.checkpoint import build_dataset, load_run


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_id(buffer: list[list[int]]) -> str:
    encoded = json.dumps(buffer, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_snapshot(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_record(
    *, model, vocab, item: dict, source_index: int, source_seed: int,
    device: str, candidate_budget: int, rollout_depth: int,
    score_batch_size: int, proposal_seed: int,
) -> dict | None:
    """Collect one trajectory; target is excluded from every policy decision."""
    prompt = item["prompt"]
    current = copy_buffer(item["buffers"][0])
    goal = copy_buffer(item["buffers"][-1])
    initial_id = _snapshot_id(current)
    problem_id = f"igsm_real_token_edit:train:{source_seed}:{source_index}"
    snapshots = [copy_buffer(current)]
    snapshot_ids = [initial_id]
    behavior_actions = []
    decisions = []
    seen = {buffer_key(current)}
    looped = False
    invalid_actions = 0
    stop_reason = "rollout_depth"
    for depth in range(rollout_depth):
        rng = random.Random(
            f"frozen-replay:{proposal_seed}:{problem_id}:{depth}"
        )
        tokens = _proposal_tokens(prompt, current, "prompt_plus_current")
        candidates = propose_deployable_edits(
            current, tokens, candidate_budget, rng
        )
        if not candidates:
            stop_reason = "empty_candidate_pool"
            break
        values = gar_scores(
            model, current, candidates, vocab.pad_id, device,
            batch_size=score_batch_size,
        )
        selected_index = max(range(len(values)), key=values.__getitem__)
        selected = candidates[selected_index]
        outcome = copy_buffer(current)
        try:
            _apply(outcome, selected)
        except (AssertionError, IndexError, ValueError):
            invalid_actions += 1
            stop_reason = "invalid_action"
            break
        behavior_actions.append(list(selected))
        snapshots.append(copy_buffer(outcome))
        state_id = _snapshot_id(outcome)
        snapshot_ids.append(state_id)
        decisions.append({
            "depth": depth + 1,
            "candidate_count": len(candidates),
            "selected_candidate_index": selected_index,
            "selected_gar_score": float(values[selected_index]),
            "proposal_seed": f"frozen-replay:{proposal_seed}:{problem_id}:{depth}",
        })
        key = buffer_key(outcome)
        if key in seen:
            looped = True
            stop_reason = "loop"
            break
        seen.add(key)
        current = outcome
    if not behavior_actions:
        return None
    return {
        "problem_id": problem_id,
        "source_split": "train",
        "source_seed": source_seed,
        "source_index": source_index,
        "snapshot_id": initial_id,
        "state_snapshot_ids": snapshot_ids,
        "prompt": prompt,
        "buffer_snapshots": snapshots,
        "behavior_actions": behavior_actions,
        "terminal_privileged_goal_buffer": goal,
        "terminal_goal_sha256": _snapshot_id(goal),
        "answer": int(item["answer"]),
        "looped": looped,
        "invalid_actions": invalid_actions,
        "stop_reason": stop_reason,
        "decisions": decisions,
        "information_regime": {
            "state_generation": "deployment_feasible_frozen_gar_greedy",
            "proposal_generation": "deployment_feasible_prompt_plus_current",
            "candidate_outcomes": "mechanically_exact_target_free",
            "goal_buffer": "terminal_privileged_training_label_only",
            "initial_buffer": "synthetic_gold_derived_corruption",
            "oracle_candidate_injection": False,
            "canonical_action_access": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    parser.add_argument("--output", "--out", dest="output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=3000)
    parser.add_argument("--candidate-budget", type=int, default=256)
    parser.add_argument("--rollout-depth", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--proposal-seed", type=int, default=3700)
    args = parser.parse_args()
    if min(
        args.examples, args.candidate_budget, args.rollout_depth,
        args.score_batch_size,
    ) < 1:
        parser.error("examples, candidate budget, depth, and batch size must be positive")

    checkpoint = Path(args.checkpoint).resolve()
    model, vocab, cfg = load_run(str(checkpoint), args.device)
    if not getattr(model, "token_aligned", False) or model.token_pred is None:
        raise ValueError("collector requires a structured token-aligned checkpoint")
    if float(cfg.objective.get("gar_action_value", {}).get("weight", 0)) <= 0:
        raise ValueError("collector requires a checkpoint with a trained GAR head")

    source_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    source_cfg.data.train_size = args.examples
    source_cfg.data.fresh_per_epoch = False
    source_cfg.data.counterfactual_k = 0
    source_cfg.data.proposal_pool_k = 0
    source_cfg.data.replay_path = None
    source_cfg.data.replay_fraction = 0.0
    source = build_dataset(source_cfg, vocab, split="train", size=args.examples)
    source_seed = int(source_cfg.data.train_seed)
    records = []
    for index in range(len(source)):
        record = collect_record(
            model=model, vocab=vocab, item=source[index], source_index=index,
            source_seed=source_seed, device=args.device,
            candidate_budget=args.candidate_budget,
            rollout_depth=args.rollout_depth,
            score_batch_size=args.score_batch_size,
            proposal_seed=args.proposal_seed,
        )
        if record is not None:
            records.append(record)

    root = Path(__file__).resolve().parents[1]
    manifest = {
        "format": REPLAY_FORMAT,
        "repository_snapshot": _git_snapshot(root),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "source_split": "train",
        "source_seed": source_seed,
        "requested_examples": args.examples,
        "collected_trajectories": len(records),
        "candidate_budget": args.candidate_budget,
        "rollout_depth": args.rollout_depth,
        "behavior_policy": "gar_greedy",
        "behavior_gar_objective": OmegaConf.to_container(
            cfg.objective.get("gar_action_value", {}), resolve=True
        ),
        "proposal_token_pool": "prompt_plus_current",
        "oracle_candidate_injection": False,
        "target_usage": (
            "stored separately for terminal-privileged training labels; never "
            "read by proposal generation, GAR scoring, or action selection"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": REPLAY_FORMAT, "manifest": manifest, "records": records}, args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({**manifest, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
