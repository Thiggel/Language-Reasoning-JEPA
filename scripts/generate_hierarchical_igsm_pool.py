#!/usr/bin/env python3
"""Generate a verified natural-language iGSM pool for fixed depth splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from textjepa.data.igsm.dataset import (
    DEFAULT_ADJECTIVES,
    DEFAULT_NOUNS,
)
from textjepa.data.igsm.graph import (
    CONST_OP,
    OPS,
    OP_WORDS,
    Problem,
    Var,
)
from textjepa.data.igsm.render import (
    prompt_sentences,
    step_sentence,
)


HELD_OUT_GRAPH_FAMILY = "query_op_mul"
HELD_OUT_TEMPLATE_FAMILY = "paraphrase_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-depth-family", type=int, default=500)
    parser.add_argument("--normal-per-depth", type=int)
    parser.add_argument("--heldout-per-depth", type=int)
    parser.add_argument("--length-per-depth", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _paraphrase_prompt(problem, rng: random.Random) -> str:
    order = list(range(len(problem.vars)))
    rng.shuffle(order)
    lines = []
    for index in order:
        variable = problem.vars[index]
        if variable.is_leaf:
            lines.append(
                f"We have {variable.const} {variable.name}."
            )
        else:
            left, right = (problem.vars[parent] for parent in variable.parents)
            lines.append(
                f"The count of {variable.name} is the count of "
                f"{left.name} {OP_WORDS[variable.op]} the count of "
                f"{right.name}."
            )
    lines.append(
        f"Determine the number of {problem.vars[problem.query].name}."
    )
    return " ".join(lines)


def _paraphrase_step(problem, index: int) -> str:
    variable = problem.vars[index]
    if variable.is_leaf:
        return f"Looking it up gives {variable.name} = {problem.values[index]}."
    left, right = variable.parents
    return (
        f"Computing {variable.name}: {problem.values[left]} "
        f"{OP_WORDS[variable.op]} {problem.values[right]} gives "
        f"{problem.values[index]}."
    )


def _state_id(problem_id: str, resolved: list[int]) -> int:
    digest = hashlib.sha256(
        f"{problem_id}:{','.join(map(str, sorted(resolved)))}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _record(
    problem,
    trace: list[int],
    *,
    problem_id: str,
    template_family: str,
    rng: random.Random,
) -> dict:
    resolved = [
        variable.idx for variable in problem.vars if variable.is_leaf
    ]
    states = [_state_id(problem_id, resolved)]
    operations = []
    for action in trace:
        variable = problem.vars[action]
        if not all(parent in resolved for parent in variable.parents):
            raise RuntimeError("constructed trace violates dependency order")
        resolved.append(action)
        text = step_sentence(problem, action)
        if template_family == HELD_OUT_TEMPLATE_FAMILY:
            text = _paraphrase_step(problem, action)
        operations.append(text)
        states.append(_state_id(problem_id, resolved))
    if problem.query not in resolved or len(trace) != len(
        [variable for variable in problem.vars if not variable.is_leaf]
    ):
        raise RuntimeError("generated symbolic trace failed verification")
    problem_text = (
        _paraphrase_prompt(problem, rng)
        if template_family == HELD_OUT_TEMPLATE_FAMILY
        else " ".join(prompt_sentences(problem, rng))
    )
    states.append(states[-1])  # final answer emission changes no symbolic state
    return {
        "corpus": "igsm",
        "problem_text": problem_text,
        "reasoning_operations": operations,
        "answer": problem.answer,
        "reasoning_depth": len(trace),
        "canonical_state_ids": states,
        "problem_id": problem_id,
        "template_family": template_family,
        "graph_family": f"query_op_{problem.vars[problem.query].op}",
        "symbolically_verified": True,
        "symbolic_trace_actions": trace,
        "query_variable": problem.query,
        "query_ancestors": sorted(problem.query_ancestors),
    }


def _exact_depth_problem(
    rng: random.Random, depth: int, *, held_out: bool
) -> Problem:
    names = rng.sample([
        (adjective, noun)
        for adjective in DEFAULT_ADJECTIVES
        for noun in DEFAULT_NOUNS
    ], depth + 2)
    variables = [
        Var(
            index, adjective, noun, CONST_OP,
            const=rng.randrange(23),
        )
        for index, (adjective, noun) in enumerate(names[:2])
    ]
    for offset, (adjective, noun) in enumerate(names[2:], start=2):
        if offset == 2:
            parents = (0, 1)
        else:
            parents = (offset - 1, rng.choice((0, 1)))
        if offset == depth + 1:
            operation = "mul" if held_out else rng.choice(("add", "sub"))
        else:
            operation = rng.choice(OPS)
        variables.append(Var(
            offset, adjective, noun, operation, parents=parents
        ))
    return Problem(tuple(variables), depth + 1, 23)


def main() -> None:
    args = parse_args()
    if args.per_depth_family < 1 or any(
        value is not None and value < 1 for value in (
            args.normal_per_depth, args.heldout_per_depth,
            args.length_per_depth,
        )
    ):
        raise ValueError("per-depth-family must be positive")
    records = []
    # normal: train/ID/length; structural: held query operation;
    # paraphrase: held renderer with non-held graph family.
    categories = ("normal", "structural", "paraphrase")
    for depth in range(2, 13):
        active = categories if depth <= 6 else ("normal",)
        targets = {
            category: (
                (args.length_per_depth or args.per_depth_family)
                if depth > 6 else
                (args.normal_per_depth or args.per_depth_family)
                if category == "normal" else
                (args.heldout_per_depth or args.per_depth_family)
            )
            for category in active
        }
        counts = {category: 0 for category in active}
        attempt = 0
        while any(counts[name] < targets[name] for name in counts):
            rng = random.Random(f"{args.seed}:{depth}:{attempt}")
            attempt += 1
            if depth > 6:
                category = "normal"
            else:
                unfinished = [
                    name for name in counts if counts[name] < targets[name]
                ]
                category = min(
                    unfinished, key=lambda name: counts[name] / targets[name]
                )
            problem = _exact_depth_problem(
                rng, depth, held_out=category == "structural"
            )
            trace = list(range(2, depth + 2))
            problem_id = (
                f"nested-igsm-s{args.seed}-d{depth}-{category}-"
                f"{counts[category]:06d}-{attempt:08d}"
            )
            template = (
                HELD_OUT_TEMPLATE_FAMILY
                if category == "paraphrase" else "canonical_v1"
            )
            records.append(_record(
                problem, trace, problem_id=problem_id,
                template_family=template, rng=rng,
            ))
            counts[category] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "count": len(records),
        "per_depth_family": args.per_depth_family,
        "normal_per_depth": args.normal_per_depth,
        "heldout_per_depth": args.heldout_per_depth,
        "length_per_depth": args.length_per_depth,
        "seed": args.seed,
        "held_out_graph_family": HELD_OUT_GRAPH_FAMILY,
        "held_out_template_family": HELD_OUT_TEMPLATE_FAMILY,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
