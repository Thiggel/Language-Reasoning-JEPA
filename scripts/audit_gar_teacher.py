"""Audit whether GAR's geometric *teacher labels* order actions correctly.

The standard counterfactual audit evaluates the trained value head and raw
one-step goal distance.  This script instead reconstructs the exact candidate
set used by the GAR objective and compares ``out.extras['ga_label']`` with the
environment's held-out post-action remaining-step order.  It therefore
separates a bad geometric teacher from failure to fit a good teacher.

Lower scores are better for both GAR distance and the learned energy.  The
symbolic remaining-step values are used only for this diagnostic and never
enter training.

Usage::

    .venv/bin/python scripts/audit_gar_teacher.py \
        --ckpt runs/disc_gar_random_h2_k2_ldad/best.pt --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from textjepa.data.igsm.dataset import collate
from textjepa.training.trainer import to_device
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, load_run


def _candidate_remaining(dataset, item: dict) -> torch.Tensor:
    """Exact post-root-action remaining steps for a raw GAR dataset item."""
    kind = item.get("ga_env_kind", "stylized")
    if kind == "faithful":
        from textjepa.data.faithful import FaithfulEnv

        problem = item.get("ga_problem")
        trace = item.get("ga_trace")
        if problem is None or trace is None:
            raise ValueError(
                "faithful random-shooting items do not retain the action "
                "trace; use a geometry-greedy faithful checkpoint"
            )
        env = FaithfulEnv(problem)
    else:
        from textjepa.data.igsm.env import SymbolicEnv

        problem, _ = dataset.problem(int(item["index"]))
        # The stylized dataset deliberately exposes the executed variable id
        # as a probe label, which is the exact trace even when n_alt consumes
        # additional RNG draws during generation.
        trace = item["var_idx"]
        env = SymbolicEnv(problem)

    t = int(item["ga_t"])
    for action in trace[:t]:
        env.step(action)
    remaining = []
    for action in item["ga_candidate_objects"]:
        child = env.clone()
        child.step(action)
        remaining.append(child.remaining_necessary())
    return torch.tensor(remaining, dtype=torch.float)


def _canonical_stylized_goals(model, dataset, items, batch, vocab):
    """EMA states for prompt + a deterministic necessary-only solution.

    This removes reference-trace distractor content while retaining only
    information available from a completed training demonstration.  Symbolic
    ancestry is used solely to construct this offline diagnostic goal.
    """
    from textjepa.data.igsm.env import SymbolicEnv

    sequences = []
    for raw in items:
        if raw.get("ga_env_kind", "stylized") != "stylized":
            return None
        problem, _ = dataset.problem(int(raw["index"]))
        env = SymbolicEnv(problem)
        steps = []
        while not env.solved:
            necessary = [
                action for action in env.feasible_actions()
                if action in problem.query_ancestors
            ]
            steps.append(vocab.encode(env.step(min(necessary))))
        sequences.append(steps)

    device = batch["prompt_tokens"].device
    bsz = len(sequences)
    n_steps = max(len(seq) for seq in sequences)
    width = max(len(step) for seq in sequences for step in seq)
    tokens = torch.full(
        (bsz, n_steps, width), vocab.pad_id,
        dtype=torch.long, device=device,
    )
    mask = torch.zeros(bsz, n_steps, dtype=torch.bool, device=device)
    for b, sequence in enumerate(sequences):
        for t, sentence in enumerate(sequence):
            tokens[b, t, : len(sentence)] = torch.tensor(
                sentence, dtype=torch.long, device=device
            )
            mask[b, t] = True
    _, states = model.encode_states(
        batch["prompt_tokens"], batch["prompt_mask"], tokens, mask,
        teacher=True,
    )
    last = mask.sum(1).clamp(min=1) - 1
    return states[torch.arange(bsz, device=device), last]


def _final_step_goals(model, batch):
    """EMA states for prompt + the demonstration's final answer sentence.

    Unlike the necessary-only diagnostic above, this construction needs no
    symbolic ancestry: the final observed sentence is identifiable directly
    from every completed demonstration.  It is therefore a deployable GAR
    goal definition rather than merely a causal audit.
    """
    step_mask = batch["step_mask"]
    bsz = len(step_mask)
    last = step_mask.sum(1).clamp(min=1) - 1
    final_tokens = batch["step_tokens"][
        torch.arange(bsz, device=step_mask.device), last
    ].unsqueeze(1)
    final_mask = torch.ones(
        bsz, 1, dtype=torch.bool, device=step_mask.device
    )
    _, states = model.encode_states(
        batch["prompt_tokens"], batch["prompt_mask"],
        final_tokens, final_mask, teacher=True,
    )
    return states[:, 0]


def _random_completion_goals(model, dataset, items, batch, vocab, count=4):
    """EMA terminal states from independent random feasible completions.

    These goal traces use no relevance annotations: at every state the policy
    chooses uniformly from the environment's feasible actions and stops once
    the query is solved.  Multiple valid completions test whether a set-valued
    goal suppresses demonstration-specific distractor content.
    """
    from textjepa.data.igsm.env import SymbolicEnv

    sequences = []
    owners = []
    for b, raw in enumerate(items):
        if raw.get("ga_env_kind", "stylized") != "stylized":
            return None
        problem, _ = dataset.problem(int(raw["index"]))
        for sample in range(count):
            rng = random.Random(f"gar-goal:{raw['index']}:{sample}")
            env = SymbolicEnv(problem)
            steps = []
            while not env.solved:
                steps.append(vocab.encode(env.step(rng.choice(env.feasible_actions()))))
            sequences.append(steps)
            owners.append(b)

    device = batch["prompt_tokens"].device
    n_steps = max(len(seq) for seq in sequences)
    width = max(len(step) for seq in sequences for step in seq)
    tokens = torch.full(
        (len(sequences), n_steps, width), vocab.pad_id,
        dtype=torch.long, device=device,
    )
    mask = torch.zeros(len(sequences), n_steps, dtype=torch.bool, device=device)
    for row, sequence in enumerate(sequences):
        for t, sentence in enumerate(sequence):
            tokens[row, t, : len(sentence)] = torch.tensor(
                sentence, dtype=torch.long, device=device
            )
            mask[row, t] = True
    owner = torch.tensor(owners, dtype=torch.long, device=device)
    _, states = model.encode_states(
        batch["prompt_tokens"].index_select(0, owner),
        batch["prompt_mask"].index_select(0, owner),
        tokens, mask, teacher=True,
    )
    last = mask.sum(1).clamp(min=1) - 1
    goals = states[torch.arange(len(sequences), device=device), last]
    return goals.reshape(len(items), count, -1)


def _random_rollout_labels(model, batch, goal):
    """Re-score the stored random-shooting leaves against another goal."""
    if goal is None or "ga_rollout_step_tokens" not in batch:
        return None
    rt = batch["ga_rollout_step_tokens"]
    rm = batch["ga_rollout_step_mask"]
    rv = batch["ga_rollout_valid"]
    bsz, candidates, rollouts, steps, width = rt.shape
    flat_mask = rm.reshape(bsz * candidates * rollouts, steps).clone()
    empty = ~flat_mask.any(1)
    flat_mask[empty, 0] = True
    _, states = model.encode_states(
        batch["prompt_tokens"].repeat_interleave(candidates * rollouts, 0),
        batch["prompt_mask"].repeat_interleave(candidates * rollouts, 0),
        rt.reshape(bsz * candidates * rollouts, steps, width),
        flat_mask,
        teacher=True,
    )
    last = flat_mask.sum(1).clamp(min=1) - 1
    leaf = states[
        torch.arange(bsz * candidates * rollouts, device=states.device), last
    ].reshape(bsz, candidates, rollouts, -1)
    distance = (
        F.layer_norm(leaf, leaf.shape[-1:])
        - F.layer_norm(goal, goal.shape[-1:]).view(bsz, 1, 1, -1)
    ).abs().mean(-1)
    distance = distance.masked_fill(~rv, float("inf"))
    return distance.amin(-1)


def _random_rollout_multigoal_labels(model, batch, goals):
    """Score stored shooting leaves against mean and nearest valid goals."""
    if goals is None or "ga_rollout_step_tokens" not in batch:
        return None, None
    rt = batch["ga_rollout_step_tokens"]
    rm = batch["ga_rollout_step_mask"]
    rv = batch["ga_rollout_valid"]
    bsz, candidates, rollouts, steps, width = rt.shape
    flat_mask = rm.reshape(bsz * candidates * rollouts, steps).clone()
    empty = ~flat_mask.any(1)
    flat_mask[empty, 0] = True
    _, states = model.encode_states(
        batch["prompt_tokens"].repeat_interleave(candidates * rollouts, 0),
        batch["prompt_mask"].repeat_interleave(candidates * rollouts, 0),
        rt.reshape(bsz * candidates * rollouts, steps, width),
        flat_mask, teacher=True,
    )
    last = flat_mask.sum(1).clamp(min=1) - 1
    leaf = states[
        torch.arange(bsz * candidates * rollouts, device=states.device), last
    ].reshape(bsz, candidates, rollouts, -1)
    ln_leaf = F.layer_norm(leaf, leaf.shape[-1:])
    mean_goal = goals.mean(1)
    mean_distance = (
        ln_leaf
        - F.layer_norm(mean_goal, mean_goal.shape[-1:]).view(bsz, 1, 1, -1)
    ).abs().mean(-1)
    set_distance = (
        ln_leaf.unsqueeze(3)
        - F.layer_norm(goals, goals.shape[-1:]).view(
            bsz, 1, 1, goals.shape[1], goals.shape[2]
        )
    ).abs().mean(-1).amin(-1)
    mean_distance = mean_distance.masked_fill(~rv, float("inf"))
    set_distance = set_distance.masked_fill(~rv, float("inf"))
    return mean_distance.amin(-1), set_distance.amin(-1)


class OrderingStats:
    """Pair and top-1 diagnostics for a lower-is-better score."""

    def __init__(self, gap: float, truth_gap: float = 0.0):
        self.gap = float(gap)
        self.truth_gap = float(truth_gap)
        self.oracle_pairs = 0
        self.decisive_oracle_pairs = 0
        self.correct = 0
        self.wrong = 0
        self.emitted_pairs = 0
        self.emitted_oracle_ties = 0
        self.top1 = 0
        self.top1_set = 0
        self.examples = 0

    def update(self, score: torch.Tensor, truth: torch.Tensor) -> None:
        score = score.detach().float().cpu()
        truth = truth.detach().float().cpu()
        n = len(score)
        self.examples += 1
        oracle_best = truth <= truth.min() + self.truth_gap
        self.top1 += int(oracle_best[score.argmin()].item())
        geometric_best = score <= score.min() + self.gap
        self.top1_set += int((oracle_best & geometric_best).any().item())
        for i in range(n):
            for j in range(i + 1, n):
                ds = float(score[i] - score[j])
                dt = float(truth[i] - truth[j])
                decisive = abs(ds) > self.gap
                oracle_tie = abs(dt) <= self.truth_gap
                if decisive:
                    self.emitted_pairs += 1
                    if oracle_tie:
                        self.emitted_oracle_ties += 1
                if oracle_tie:
                    continue
                self.oracle_pairs += 1
                if not decisive:
                    continue
                self.decisive_oracle_pairs += 1
                if (ds < 0) == (dt < 0):
                    self.correct += 1
                else:
                    self.wrong += 1

    def summary(self) -> dict[str, float | int]:
        div = lambda x, n: float(x / max(n, 1))
        return {
            "examples": self.examples,
            "oracle_pairs": self.oracle_pairs,
            "oracle_pair_coverage": div(
                self.decisive_oracle_pairs, self.oracle_pairs
            ),
            "oracle_pair_accuracy_decisive": div(
                self.correct, self.decisive_oracle_pairs
            ),
            # Tied/filtered teacher pairs receive zero credit here.
            "oracle_pair_accuracy_all": div(self.correct, self.oracle_pairs),
            "oracle_pair_tau_a": div(
                self.correct - self.wrong, self.oracle_pairs
            ),
            "emitted_pairs": self.emitted_pairs,
            "emitted_oracle_tie_rate": div(
                self.emitted_oracle_ties, self.emitted_pairs
            ),
            "top1_accuracy": div(self.top1, self.examples),
            "top1_set_overlap": div(self.top1_set, self.examples),
        }


@torch.no_grad()
def audit(args) -> dict:
    device = torch.device(args.device)
    model, vocab, cfg = load_run(args.ckpt, args.device)
    if model.__class__.__name__ != "DiscourseJEPA":
        raise TypeError("GAR teacher audit requires a DiscourseJEPA checkpoint")
    if int(cfg.data.get("geo_rank_k", 0)) <= 0:
        raise ValueError("checkpoint was not trained with GAR candidates")
    if args.horizon is not None:
        if args.horizon < 1:
            raise ValueError("horizon must be positive")
        cfg.data.geo_rank_horizon = args.horizon
        cfg.data.geo_rank_policy = "greedy"
    if args.beam_width is not None:
        if args.beam_width < 1:
            raise ValueError("beam width must be positive")
        cfg.data.geo_rank_beam_width = args.beam_width
        cfg.data.geo_rank_policy = "greedy"
    dataset = build_dataset(
        cfg, vocab, split=args.split,
        size=max(args.n_anchors * 4, args.n_anchors),
    )
    label_gap = float(cfg.objective.geo_rank.get("label_gap", 0.02))
    teacher = OrderingStats(label_gap)
    sweep_gaps = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)
    teacher_sweep = {gap: OrderingStats(gap) for gap in sweep_gaps}
    teacher_strata = {
        "trace_without_distractors": OrderingStats(label_gap),
        "trace_with_distractors": OrderingStats(label_gap),
        "anchor_before_any_distractor": OrderingStats(label_gap),
        "anchor_after_a_distractor": OrderingStats(label_gap),
    }
    canonical_teacher = OrderingStats(label_gap)
    canonical_strata = {
        "trace_without_distractors": OrderingStats(label_gap),
        "trace_with_distractors": OrderingStats(label_gap),
    }
    final_step_teacher = OrderingStats(label_gap)
    final_step_strata = {
        "trace_without_distractors": OrderingStats(label_gap),
        "trace_with_distractors": OrderingStats(label_gap),
    }
    random_goal_mean_teacher = OrderingStats(label_gap)
    random_goal_set_teacher = OrderingStats(label_gap)
    student = OrderingStats(0.0)
    teacher_student = OrderingStats(0.0, truth_gap=label_gap)
    processed = 0

    pending: list[dict] = []
    for index in range(len(dataset)):
        item = dataset[index]
        if "ga_t" not in item:
            continue
        pending.append(item)
        if len(pending) < args.batch_size and processed + len(pending) < args.n_anchors:
            continue
        batch = to_device(collate(pending, vocab.pad_id), device)
        out = model(batch)
        labels = out.extras["ga_label"]
        energies = out.extras["ga_energy"]
        valid = out.extras["ga_valid"]
        if "ga_rollout_step_tokens" in batch:
            canonical_goal = _canonical_stylized_goals(
                model, dataset, pending, batch, vocab
            )
            canonical_labels = _random_rollout_labels(
                model, batch, canonical_goal
            )
            final_step_labels = _random_rollout_labels(
                model, batch, _final_step_goals(model, batch)
            )
            random_goals = _random_completion_goals(
                model, dataset, pending, batch, vocab
            )
            random_goal_mean_labels, random_goal_set_labels = (
                _random_rollout_multigoal_labels(model, batch, random_goals)
            )
        else:
            canonical_labels = final_step_labels = None
            random_goal_mean_labels = random_goal_set_labels = None
        for row, raw in enumerate(pending):
            n = len(raw["ga_candidate_objects"])
            row_valid = valid[row, :n]
            truth = _candidate_remaining(dataset, raw)[row_valid.cpu()]
            geometry = labels[row, :n][row_valid]
            energy = energies[row, :n][row_valid]
            if len(truth) < 2:
                continue
            teacher.update(geometry, truth)
            for stats in teacher_sweep.values():
                stats.update(geometry, truth)
            necessary = raw.get("necessary", [])
            t = int(raw["ga_t"])
            trace_key = (
                "trace_with_distractors"
                if any(not bool(x) for x in necessary)
                else "trace_without_distractors"
            )
            prefix_key = (
                "anchor_after_a_distractor"
                if any(not bool(x) for x in necessary[:t])
                else "anchor_before_any_distractor"
            )
            teacher_strata[trace_key].update(geometry, truth)
            teacher_strata[prefix_key].update(geometry, truth)
            if canonical_labels is not None:
                canonical_geometry = canonical_labels[row, :n][row_valid]
                canonical_teacher.update(canonical_geometry, truth)
                canonical_strata[trace_key].update(canonical_geometry, truth)
            if final_step_labels is not None:
                final_geometry = final_step_labels[row, :n][row_valid]
                final_step_teacher.update(final_geometry, truth)
                final_step_strata[trace_key].update(final_geometry, truth)
            if random_goal_mean_labels is not None:
                random_goal_mean_teacher.update(
                    random_goal_mean_labels[row, :n][row_valid], truth
                )
                random_goal_set_teacher.update(
                    random_goal_set_labels[row, :n][row_valid], truth
                )
            student.update(energy, truth)
            # Treat teacher distance as the lower-is-better reference to
            # quantify whether the value head fitted the labels it received.
            teacher_student.update(energy, geometry.detach().cpu())
            processed += 1
            if processed >= args.n_anchors:
                break
        pending = []
        if processed >= args.n_anchors:
            break

    result = {
        "checkpoint": args.ckpt,
        "split": args.split,
        "gar_policy": str(cfg.data.get("geo_rank_policy", "random")),
        "gar_horizon": int(cfg.data.get("geo_rank_horizon", 1)),
        "gar_beam_width": int(cfg.data.get("geo_rank_beam_width", 1)),
        "gar_candidates": int(cfg.data.get("geo_rank_k", 0)) + 1,
        "label_gap": label_gap,
        "teacher_vs_oracle": teacher.summary(),
        "teacher_gap_sweep": {
            f"{gap:g}": stats.summary()
            for gap, stats in teacher_sweep.items()
        },
        "teacher_strata": {
            name: stats.summary() for name, stats in teacher_strata.items()
        },
        "canonical_goal_teacher_vs_oracle": canonical_teacher.summary(),
        "canonical_goal_teacher_strata": {
            name: stats.summary() for name, stats in canonical_strata.items()
        },
        "final_step_goal_teacher_vs_oracle": final_step_teacher.summary(),
        "final_step_goal_teacher_strata": {
            name: stats.summary() for name, stats in final_step_strata.items()
        },
        "random_completion_mean_goal_teacher_vs_oracle": (
            random_goal_mean_teacher.summary()
        ),
        "random_completion_set_goal_teacher_vs_oracle": (
            random_goal_set_teacher.summary()
        ),
        "student_vs_oracle": student.summary(),
        "student_vs_teacher": teacher_student.summary(),
    }
    return result


def _normalize_cli(argv: list[str]) -> list[str]:
    """Accept the legacy Hydra-style arguments used by older job chains."""
    aliases = {
        "ckpt": "--ckpt",
        "device": "--device",
        "split": "--split",
        "n_episodes": "--n-anchors",
        "n_anchors": "--n-anchors",
        "batch_size": "--batch-size",
        "beam_width": "--beam-width",
        "seed": "--seed",
        "out": "--out",
    }
    normalized: list[str] = []
    for token in argv:
        if "=" in token and not token.startswith("--"):
            key, value = token.split("=", 1)
            option = aliases.get(key)
            if option is not None:
                normalized.extend([option, value])
                continue
        normalized.append(token)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--n-anchors", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--horizon", type=int,
        help=("recompute the geometric continuation teacher at this horizon "
              "while holding the checkpoint fixed"),
    )
    parser.add_argument(
        "--beam-width", type=int,
        help=("retain this many lowest-distance continuations per root; "
              "one recovers the original geometry-greedy teacher"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args(_normalize_cli(sys.argv[1:]))
    seed_everything(args.seed)
    result = audit(args)
    destination = (
        Path(args.out) if args.out
        else Path(args.ckpt).parent / "gar_teacher_audit.json"
    )
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
