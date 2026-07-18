"""Small closed-loop planning evaluation for structured faithful token edits.

The two deployable proposal pools use only the observed current buffer, or
the prompt plus current buffer.  The clean target is used only for evaluation.
An additional oracle-injected condition is explicitly candidate privileged.

Example:
    .venv/bin/python scripts/plan_faithful_token_edits.py \
        --ckpt runs/edit_token_structured_gar_h1/best.pt --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from textjepa.data.faithful_token_edits import OPS, _apply
from textjepa.utils.checkpoint import build_dataset, load_run


Edit = tuple[str, int, int | None]
Buffer = list[list[int]]
ScoreFunction = Callable[[Buffer, list[Edit]], list[float]]


def copy_buffer(buffer: Buffer) -> Buffer:
    return [list(sentence) for sentence in buffer]


def buffer_key(buffer: Buffer) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(sentence) for sentence in buffer)


def flatten(buffer: Buffer) -> list[int]:
    return [token for sentence in buffer for token in sentence]


def token_edit_distance(left: list[int], right: list[int]) -> int:
    """Memory-bounded Levenshtein distance over token ids."""
    previous = list(range(len(right) + 1))
    for row, left_token in enumerate(left, start=1):
        current = [row]
        for column, right_token in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + int(left_token != right_token),
            ))
        previous = current
    return previous[-1]


def buffer_distance(left: Buffer, right: Buffer) -> int:
    """Boundary-preserving edit distance, summed over official steps."""
    if len(left) != len(right):
        raise ValueError("planning buffers must preserve official step count")
    return sum(token_edit_distance(a, b) for a, b in zip(left, right))


def canonical_oracle_edit(current: Buffer, target: Buffer) -> Edit | None:
    """Return one target-privileged edit on a shortest boundary-safe path."""
    if len(current) != len(target):
        raise ValueError("planning buffers must preserve official step count")
    offset = 0
    for sentence, goal in zip(current, target):
        if sentence == goal:
            offset += len(sentence)
            continue
        n, m = len(sentence), len(goal)
        suffix = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n, -1, -1):
            suffix[i][m] = n - i
        for j in range(m, -1, -1):
            suffix[n][j] = m - j
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                suffix[i][j] = min(
                    1 + suffix[i + 1][j],
                    1 + suffix[i][j + 1],
                    int(sentence[i] != goal[j]) + suffix[i + 1][j + 1],
                )
        i = j = 0
        while i < n and j < m and sentence[i] == goal[j]:
            i += 1
            j += 1
        distance = suffix[i][j]
        choices = []
        if i < n and j < m and 1 + suffix[i + 1][j + 1] == distance:
            choices.append(("replace", offset + i, goal[j]))
        if j < m and 1 + suffix[i][j + 1] == distance:
            choices.append(("insert", offset + i, goal[j]))
        if i < n and len(sentence) > 1 and 1 + suffix[i + 1][j] == distance:
            choices.append(("delete", offset + i, None))
        before = buffer_distance(current, target)
        for action in choices:
            outcome = copy_buffer(current)
            try:
                _apply(outcome, action)
            except (AssertionError, IndexError, ValueError):
                continue
            if buffer_distance(outcome, target) == before - 1:
                return action
        raise RuntimeError("no representable boundary-safe shortest-path edit")
    return None


def proposal_tokens(prompt: list[list[int]], buffer: Buffer, pool: str) -> list[int]:
    if pool == "current_buffer":
        source = flatten(buffer)
    elif pool == "prompt_plus_current":
        source = flatten(prompt) + flatten(buffer)
    else:
        raise ValueError(f"unknown proposal pool: {pool}")
    return list(dict.fromkeys(source))


def propose_edits(
    buffer: Buffer, tokens: list[int], max_candidates: int, rng: random.Random
) -> list[Edit]:
    """Build an operation-balanced, target-free bounded candidate set."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    current = flatten(buffer)
    deletes, inserts, replaces = [], [], []
    offset = 0
    for sentence in buffer:
        # Match the data contract: never delete a step-final token, so its
        # inverse insertion remains unambiguous under flattened pointers.
        if len(sentence) > 1:
            deletes.extend(("delete", position, None) for position in range(
                offset, offset + len(sentence) - 1
            ))
        offset += len(sentence)
    for position in range(len(current) + 1):
        inserts.extend(("insert", position, token) for token in tokens)
    for position, old in enumerate(current):
        replaces.extend(
            ("replace", position, token) for token in tokens if token != old
        )
    groups = [deletes, inserts, replaces]
    for group in groups:
        rng.shuffle(group)
    selected = []
    while len(selected) < max_candidates and any(groups):
        for group in groups:
            if group and len(selected) < max_candidates:
                selected.append(group.pop())
    # Different operation paths can only duplicate if the token source did.
    return list(dict.fromkeys(selected))


def pad_token_state_for_insertions(
    states: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reserve the N+1 state slot required by a literal insertion."""
    return F.pad(states, (0, 0, 0, 1)), F.pad(mask, (0, 1), value=False)


def _buffer_tensor(buffer: Buffer, pad_id: int, device: str) -> torch.Tensor:
    width = max(max((len(sentence) for sentence in buffer), default=1), 1)
    tokens = torch.full(
        (1, 1, max(len(buffer), 1), width), pad_id,
        dtype=torch.long, device=device,
    )
    for index, sentence in enumerate(buffer):
        tokens[0, 0, index, :len(sentence)] = torch.tensor(
            sentence, dtype=torch.long, device=device
        )
    return tokens


@torch.no_grad()
def gar_scores(
    model, buffer: Buffer, candidates: list[Edit], pad_id: int,
    device: str, batch_size: int = 512,
) -> list[float]:
    """Score V(s,a) after re-encoding the observed current buffer."""
    raw = _buffer_tensor(buffer, pad_id, device)
    states, mask = model.encode_token_buffers(raw, mode="online")
    states, mask = pad_token_state_for_insertions(states[:, 0], mask[:, 0])
    pooled = model._pool_tokens(states, mask)
    scores = []
    inverse_ops = {name: index for name, index in OPS.items()}
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        count = len(chunk)
        operations = torch.tensor(
            [inverse_ops[action[0]] for action in chunk], device=device
        )
        positions = torch.tensor([action[1] for action in chunk], device=device)
        content_ids = torch.tensor([
            pad_id if action[2] is None else action[2] for action in chunk
        ], device=device)
        content = model.chunk_encoder.tok(content_ids)
        action = model.token_pred.encode_action(
            states.expand(count, -1, -1), mask.expand(count, -1),
            operations, positions, content,
        )
        value = model.gar_head(torch.cat([
            pooled.expand(count, -1), action
        ], dim=-1)).squeeze(-1)
        scores.extend(float(item) for item in value.cpu())
    return scores


@dataclass
class EpisodeMetrics:
    initial_distance: int
    final_distance: int = 0
    decisions: int = 0
    source_ceiling_hits: int = 0
    bounded_recall_hits: int = 0
    selected_advantages: list[int] = field(default_factory=list)
    invalid_actions: int = 0
    looped: bool = False
    recovered: bool = False
    oracle_injections: int = 0


def run_episode(
    prompt: list[list[int]], initial: Buffer, target: Buffer, pool: str,
    policy: str, score: ScoreFunction | None, rng: random.Random,
    max_candidates: int, max_steps: int, inject_oracle: bool = False,
    selection_rng: random.Random | None = None,
) -> EpisodeMetrics:
    """Run target-free proposals with receding observation after every edit."""
    current = copy_buffer(initial)
    selection_rng = selection_rng or rng
    initial_distance = buffer_distance(current, target)
    metrics = EpisodeMetrics(initial_distance=initial_distance)
    seen = {buffer_key(current)}
    for _ in range(max_steps):
        oracle = canonical_oracle_edit(current, target)
        if oracle is None:
            metrics.recovered = True
            break
        tokens = proposal_tokens(prompt, current, pool)
        candidates = propose_edits(current, tokens, max_candidates, rng)
        metrics.decisions += 1
        source_hit = oracle[2] is None or oracle[2] in tokens
        metrics.source_ceiling_hits += int(source_hit)
        metrics.bounded_recall_hits += int(oracle in candidates)
        if inject_oracle and oracle not in candidates:
            candidates.append(oracle)
            metrics.oracle_injections += 1
        if not candidates:
            break
        if policy == "random":
            selected = selection_rng.choice(candidates)
        elif policy == "gar_greedy":
            if score is None:
                raise ValueError("gar_greedy requires a score function")
            values = score(current, candidates)
            if len(values) != len(candidates):
                raise ValueError("score function returned the wrong number of values")
            selected = candidates[max(range(len(values)), key=values.__getitem__)]
        else:
            raise ValueError(f"unknown policy: {policy}")
        before = buffer_distance(current, target)
        try:
            _apply(current, selected)
        except (AssertionError, IndexError, ValueError):
            metrics.invalid_actions += 1
            break
        after = buffer_distance(current, target)
        metrics.selected_advantages.append(before - after)
        key = buffer_key(current)
        if key in seen:
            metrics.looped = True
            break
        seen.add(key)
    metrics.recovered = current == target
    metrics.final_distance = buffer_distance(current, target)
    return metrics


def aggregate(episodes: list[EpisodeMetrics]) -> dict:
    decisions = sum(item.decisions for item in episodes)
    selected = [value for item in episodes for value in item.selected_advantages]
    initial = sum(item.initial_distance for item in episodes)
    final = sum(item.final_distance for item in episodes)
    source_ceiling = (
        sum(item.source_ceiling_hits for item in episodes) / max(decisions, 1)
    )
    bounded_recall = (
        sum(item.bounded_recall_hits for item in episodes) / max(decisions, 1)
    )
    return {
        "episodes": len(episodes),
        "decisions": decisions,
        "proposal_recall_ceiling": source_ceiling,
        "proposal_coverage": bounded_recall,
        "proposal_source_token_recall_ceiling": source_ceiling,
        "bounded_canonical_oracle_edit_recall": bounded_recall,
        "selected_true_advantage_mean": (
            sum(selected) / len(selected) if selected else None
        ),
        "normalized_edit_distance_improvement": (
            sum(
                (item.initial_distance - item.final_distance)
                / max(item.initial_distance, 1)
                for item in episodes
            ) / max(len(episodes), 1)
        ),
        "distance_weighted_normalized_edit_distance_improvement": (
            (initial - final) / max(initial, 1)
        ),
        "exact_recovery_rate": (
            sum(item.recovered for item in episodes) / max(len(episodes), 1)
        ),
        "loop_rate": (
            sum(item.looped for item in episodes) / max(len(episodes), 1)
        ),
        "invalid_action_rate": (
            sum(item.invalid_actions for item in episodes)
            / max(sum(len(item.selected_advantages) + item.invalid_actions for item in episodes), 1)
        ),
        "oracle_injections": sum(item.oracle_injections for item in episodes),
        "mean_initial_edit_distance": initial / max(len(episodes), 1),
        "mean_final_edit_distance": final / max(len(episodes), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.examples, args.max_candidates, args.max_steps, args.score_batch_size) < 1:
        parser.error("examples, candidates, steps, and score batch size must be positive")

    model, vocab, cfg = load_run(args.ckpt, args.device)
    if not getattr(model, "token_aligned", False) or model.token_pred is None:
        raise ValueError("planning requires a structured token-aligned edit checkpoint")
    gar_cfg = cfg.objective.get("gar_action_value", {})
    if float(gar_cfg.get("weight", 0.0)) <= 0:
        raise ValueError("checkpoint has no trained GAR action-value objective")
    dataset = build_dataset(cfg, vocab, "test", size=args.examples)
    raw_items = [dataset[index] for index in range(len(dataset))]

    conditions = [
        ("current_buffer_random", "current_buffer", "random", False),
        ("current_buffer_gar_greedy", "current_buffer", "gar_greedy", False),
        ("prompt_plus_current_random", "prompt_plus_current", "random", False),
        ("prompt_plus_current_gar_greedy", "prompt_plus_current", "gar_greedy", False),
        (
            "candidate_privileged_oracle_injected_gar_greedy",
            "prompt_plus_current", "gar_greedy", True,
        ),
    ]
    results = {}
    for name, pool, policy, inject in conditions:
        episodes = []
        for index, item in enumerate(raw_items):
            scorer = None
            if policy == "gar_greedy":
                scorer = lambda current, candidates: gar_scores(
                    model, current, candidates, vocab.pad_id, args.device,
                    args.score_batch_size,
                )
            episodes.append(run_episode(
                item["prompt"], item["buffers"][0], item["buffers"][-1],
                pool, policy, scorer,
                random.Random(
                    f"faithful-edit-proposals:{args.seed}:{pool}:{index}"
                ),
                args.max_candidates, args.max_steps, inject,
                random.Random(
                    f"faithful-edit-selection:{args.seed}:{name}:{index}"
                ),
            ))
        results[name] = {
            "candidate_pool": pool,
            "policy": policy,
            "information_regime": (
                "candidate_privileged_oracle_injected_diagnostic"
                if inject else "deployment_feasible_target_free"
            ),
            **aggregate(episodes),
        }
    payload = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "split": "test",
        "candidate_budget": args.max_candidates,
        "maximum_receding_steps": args.max_steps,
        "target_usage": (
            "metrics and separately labelled oracle injection only; deployable "
            "candidate pools and GAR scores do not access the target"
        ),
        "boundary_contract": "official nested step boundaries preserved",
        "insertion_state_capacity": "current token state padded from N to N+1",
        "conditions": results,
    }
    destination = args.out or Path(args.ckpt).parent / "faithful_token_edit_planning.json"
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
