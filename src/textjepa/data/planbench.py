"""Executable PlanBench Blocksworld adapter for observed-action JEPA.

The adapter parses the official four-operator PDDL instances, solves them by
shortest-path search, and emits domain-neutral observed-action episodes.  The
public catalogue contains every grounded action.  Feasibility and optimal
continuations are used only for training labels/teacher rollouts and are not
part of a deployed policy observation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import re

from textjepa.data.observed_action import (
    Counterfactual,
    ObservedActionEpisode,
    ObservedTransition,
)

Atom = tuple[str, ...]
State = frozenset[Atom]


@dataclass(frozen=True, order=True)
class BlocksAction:
    operator: str
    block: str
    support: str | None = None

    @property
    def text(self) -> str:
        if self.operator == "pick-up":
            return f"pick up block {self.block} from the table"
        if self.operator == "put-down":
            return f"put down block {self.block} on the table"
        if self.operator == "stack":
            return f"stack block {self.block} on block {self.support}"
        if self.operator == "unstack":
            return f"unstack block {self.block} from block {self.support}"
        raise ValueError(f"unknown Blocksworld operator: {self.operator}")


@dataclass(frozen=True)
class BlocksProblem:
    problem_id: str
    objects: tuple[str, ...]
    initial: State
    goal: State


def _section(text: str, name: str, next_names: tuple[str, ...]) -> str:
    lower = text.lower()
    start = lower.find(name)
    if start < 0:
        raise ValueError(f"missing PDDL section {name}")
    end = len(text)
    for candidate in next_names:
        position = lower.find(candidate, start + len(name))
        if position >= 0:
            end = min(end, position)
    return text[start:end]


def _atoms(section: str) -> State:
    found = []
    for match in re.finditer(
        r"\((handempty|clear|ontable|holding|on)\s*([^()]*)\)",
        section,
        flags=re.IGNORECASE,
    ):
        predicate = match.group(1).lower()
        arguments = tuple(match.group(2).lower().split())
        found.append((predicate, *arguments))
    return frozenset(found)


def parse_blocksworld_pddl(text: str, problem_id: str | None = None) -> BlocksProblem:
    identity = problem_id
    if identity is None:
        match = re.search(r"\(problem\s+([^\s)]+)", text, re.IGNORECASE)
        if not match:
            raise ValueError("missing PDDL problem identifier")
        identity = match.group(1)
    object_section = _section(text, "(:objects", ("(:init",))
    objects = tuple(re.findall(r"[a-z0-9_-]+", object_section, re.IGNORECASE)[1:])
    if not objects:
        raise ValueError("Blocksworld problem has no objects")
    initial = _atoms(_section(text, "(:init", ("(:goal",)))
    goal = _atoms(_section(text, "(:goal", ()))
    if not initial or not goal:
        raise ValueError("Blocksworld initial state and goal must be non-empty")
    return BlocksProblem(str(identity), objects, initial, goal)


def load_blocksworld_pddl(path: str | Path) -> BlocksProblem:
    path = Path(path)
    return parse_blocksworld_pddl(path.read_text(), problem_id=path.stem)


def action_catalogue(objects: tuple[str, ...]) -> tuple[BlocksAction, ...]:
    actions = []
    for block in objects:
        actions.extend([
            BlocksAction("pick-up", block),
            BlocksAction("put-down", block),
        ])
        for support in objects:
            if block != support:
                actions.extend([
                    BlocksAction("stack", block, support),
                    BlocksAction("unstack", block, support),
                ])
    return tuple(actions)


def is_applicable(state: State, action: BlocksAction) -> bool:
    block, support = action.block, action.support
    if action.operator == "pick-up":
        return all(atom in state for atom in (
            ("clear", block), ("ontable", block), ("handempty",)
        ))
    if action.operator == "put-down":
        return ("holding", block) in state
    if action.operator == "stack":
        return (
            ("holding", block) in state
            and ("clear", str(support)) in state
        )
    if action.operator == "unstack":
        return all(atom in state for atom in (
            ("on", block, str(support)),
            ("clear", block),
            ("handempty",),
        ))
    raise ValueError(f"unknown Blocksworld operator: {action.operator}")


def transition(state: State, action: BlocksAction) -> State:
    if not is_applicable(state, action):
        raise ValueError(f"inapplicable action: {action.text}")
    add: set[Atom]
    delete: set[Atom]
    block, support = action.block, str(action.support)
    if action.operator == "pick-up":
        add = {("holding", block)}
        delete = {("clear", block), ("ontable", block), ("handempty",)}
    elif action.operator == "put-down":
        add = {("clear", block), ("handempty",), ("ontable", block)}
        delete = {("holding", block)}
    elif action.operator == "stack":
        add = {("handempty",), ("clear", block), ("on", block, support)}
        delete = {("clear", support), ("holding", block)}
    else:
        add = {("holding", block), ("clear", support)}
        delete = {
            ("on", block, support), ("clear", block), ("handempty",)
        }
    return frozenset((set(state) - delete) | add)


def goal_reached(state: State, goal: State) -> bool:
    return goal.issubset(state)


def shortest_plan(
    start: State,
    goal: State,
    catalogue: tuple[BlocksAction, ...],
    max_expansions: int = 200_000,
) -> tuple[BlocksAction, ...] | None:
    if goal_reached(start, goal):
        return ()
    frontier = deque([start])
    parent: dict[State, tuple[State, BlocksAction] | None] = {start: None}
    solved = None
    while frontier and len(parent) <= max_expansions:
        state = frontier.popleft()
        for action in catalogue:
            if not is_applicable(state, action):
                continue
            child = transition(state, action)
            if child in parent:
                continue
            parent[child] = (state, action)
            if goal_reached(child, goal):
                solved = child
                frontier.clear()
                break
            frontier.append(child)
    if solved is None:
        return None
    plan = []
    cursor = solved
    while parent[cursor] is not None:
        previous, action = parent[cursor]
        plan.append(action)
        cursor = previous
    return tuple(reversed(plan))


def render_state(state: State) -> str:
    clauses = []
    for atom in sorted(state):
        if atom[0] == "handempty":
            clauses.append("the hand is empty")
        elif atom[0] == "holding":
            clauses.append(f"the hand holds block {atom[1]}")
        elif atom[0] == "clear":
            clauses.append(f"block {atom[1]} is clear")
        elif atom[0] == "ontable":
            clauses.append(f"block {atom[1]} is on the table")
        elif atom[0] == "on":
            clauses.append(f"block {atom[1]} is on block {atom[2]}")
    return "; ".join(clauses) + " ."


def render_goal(goal: State) -> str:
    return "goal: " + render_state(goal)


def compile_blocksworld_episode(
    problem: BlocksProblem,
    split: str,
    teacher_horizon: int = 8,
    counterfactual_k: int | None = None,
) -> ObservedActionEpisode:
    catalogue = action_catalogue(problem.objects)
    expert = shortest_plan(problem.initial, problem.goal, catalogue)
    if expert is None or not expert:
        raise ValueError(f"problem {problem.problem_id} has no non-empty plan")
    catalogue_text = tuple(action.text for action in catalogue)
    observed = []
    state = problem.initial
    for executed in expert:
        feasible = [
            action for action in catalogue
            if action != executed and is_applicable(state, action)
        ]
        if counterfactual_k is not None:
            feasible = feasible[:max(0, int(counterfactual_k))]
        counterfactuals = []
        for alternative in feasible:
            next_state = transition(state, alternative)
            continuation = shortest_plan(next_state, problem.goal, catalogue)
            rollout_states = []
            cursor = next_state
            if continuation is not None:
                for action in continuation[:max(teacher_horizon - 1, 0)]:
                    cursor = transition(cursor, action)
                    rollout_states.append(render_state(cursor))
            counterfactuals.append(Counterfactual(
                alternative.text,
                render_state(next_state),
                (tuple(rollout_states),),
            ))
        next_state = transition(state, executed)
        observed.append(ObservedTransition(
            executed.text,
            render_state(next_state),
            catalogue_text,
            tuple(
                action.text for action in catalogue
                if is_applicable(state, action)
            ),
            tuple(counterfactuals),
        ))
        state = next_state
    if not goal_reached(state, problem.goal):
        raise AssertionError("compiled expert trajectory does not reach goal")
    return ObservedActionEpisode(
        episode_id=f"planbench-blocksworld-{problem.problem_id}",
        domain="planbench-blocksworld",
        split=split,
        prompt=(render_state(problem.initial), render_goal(problem.goal)),
        goal=render_goal(problem.goal),
        transitions=tuple(observed),
        metadata={
            "objects": len(problem.objects),
            "optimal_plan_length": len(expert),
            "symbolic_environment": True,
            "environment_spec": {
                "objects": list(problem.objects),
                "initial": [list(atom) for atom in sorted(problem.initial)],
                "goal": [list(atom) for atom in sorted(problem.goal)],
            },
        },
    )
