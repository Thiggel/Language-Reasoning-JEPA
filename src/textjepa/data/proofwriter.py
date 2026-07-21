"""Executable ProofWriter Datalog adapter.

Natural-language facts and rules remain the model inputs.  Formal
representations supplied by ProofWriter are used only by this environment to
enumerate legal rule applications, execute them, and verify query completion.
That symbolic machinery is therefore an environment/reference, never a model
feature or a paper-facing non-symbolic supervision claim.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re

from textjepa.data.observed_action import (
    Counterfactual,
    ObservedActionEpisode,
    ObservedTransition,
)

Fact = tuple[str, ...]
VARIABLES = {"someone", "something"}
_GROUP = re.compile(r'\((?:"[^"]+"\s*)+\)')
_QUOTED = re.compile(r'"([^"]+)"')


def parse_fact(representation: str) -> Fact:
    groups = _GROUP.findall(representation)
    if len(groups) != 1:
        raise ValueError(f"expected one fact representation: {representation}")
    fact = tuple(_QUOTED.findall(groups[0]))
    if len(fact) < 4 or fact[-1] not in {"+", "-"}:
        raise ValueError(f"invalid ProofWriter fact: {representation}")
    return fact


@dataclass(frozen=True)
class ProofRule:
    rule_id: str
    text: str
    antecedents: tuple[Fact, ...]
    conclusion: Fact


def parse_rule(rule_id: str, item: dict) -> ProofRule:
    groups = [tuple(_QUOTED.findall(group)) for group in _GROUP.findall(
        item["representation"]
    )]
    if len(groups) < 2 or "->" not in item["representation"]:
        raise ValueError(f"invalid ProofWriter rule {rule_id}")
    return ProofRule(rule_id, str(item["text"]), tuple(groups[:-1]), groups[-1])


@dataclass(frozen=True)
class RuleApplication:
    rule: ProofRule
    premises: tuple[Fact, ...]
    conclusion: Fact

    @property
    def text(self) -> str:
        premises = " ; ".join(render_fact(value) for value in self.premises)
        return f"apply rule: {self.rule.text} using facts: {premises}"


def _match(pattern: Fact, fact: Fact, bindings: dict[str, str]) -> dict[str, str] | None:
    if len(pattern) != len(fact):
        return None
    result = dict(bindings)
    for expected, observed in zip(pattern, fact):
        if expected.lower() in VARIABLES:
            key = expected.lower()
            if key in result and result[key] != observed:
                return None
            result[key] = observed
        elif expected != observed:
            return None
    return result


def _instantiate(pattern: Fact, bindings: dict[str, str]) -> Fact:
    result = []
    for token in pattern:
        if token.lower() in VARIABLES:
            if token.lower() not in bindings:
                raise ValueError("unbound variable in rule conclusion")
            result.append(bindings[token.lower()])
        else:
            result.append(token)
    return tuple(result)


def rule_applications(
    facts: frozenset[Fact], rules: tuple[ProofRule, ...]
) -> tuple[RuleApplication, ...]:
    applications = []
    ordered_facts = tuple(sorted(facts))
    for rule in rules:
        partial = [(dict(), tuple())]
        for antecedent in rule.antecedents:
            next_partial = []
            for bindings, premises in partial:
                for fact in ordered_facts:
                    matched = _match(antecedent, fact, bindings)
                    if matched is not None:
                        next_partial.append((matched, premises + (fact,)))
            partial = next_partial
        for bindings, premises in partial:
            conclusion = _instantiate(rule.conclusion, bindings)
            if conclusion not in facts:
                applications.append(RuleApplication(rule, premises, conclusion))
    # Multiple syntactically identical paths are not distinct policy actions.
    unique = {application.text: application for application in applications}
    return tuple(unique[key] for key in sorted(unique))


def render_fact(fact: Fact) -> str:
    subject, relation, obj, polarity = fact
    negative = polarity == "-"
    if relation == "is":
        return f"{subject} is {'not ' if negative else ''}{obj}."
    return f"{subject} does {'not ' if negative else ''}{relation} {obj}."


def shortest_derivation(
    initial: frozenset[Fact],
    rules: tuple[ProofRule, ...],
    target: Fact,
    max_expansions: int = 100_000,
) -> tuple[RuleApplication, ...] | None:
    if target in initial:
        return ()
    frontier = deque([initial])
    parent: dict[
        frozenset[Fact], tuple[frozenset[Fact], RuleApplication] | None
    ] = {initial: None}
    solved = None
    while frontier and len(parent) <= max_expansions:
        state = frontier.popleft()
        for application in rule_applications(state, rules):
            child = state | {application.conclusion}
            if child in parent:
                continue
            parent[child] = (state, application)
            if target in child:
                solved = child
                frontier.clear()
                break
            frontier.append(child)
    if solved is None:
        return None
    derivation = []
    cursor = solved
    while parent[cursor] is not None:
        previous, application = parent[cursor]
        derivation.append(application)
        cursor = previous
    return tuple(reversed(derivation))


def _target_for_question(question: dict) -> Fact | None:
    answer = question.get("answer")
    if answer not in {True, False}:
        return None
    fact = parse_fact(question["representation"])
    if answer:
        return fact
    return (*fact[:-1], "+" if fact[-1] == "-" else "-")


def compile_proofwriter_episode(
    record: dict,
    question_id: str,
    split: str,
    teacher_horizon: int = 8,
    counterfactual_k: int | None = None,
) -> ObservedActionEpisode:
    question = record["questions"][question_id]
    target = _target_for_question(question)
    if target is None:
        raise ValueError("unknown ProofWriter questions are not trajectory tasks")
    initial = frozenset(
        parse_fact(item["representation"])
        for item in record["triples"].values()
    )
    rules = tuple(
        parse_rule(rule_id, item) for rule_id, item in record["rules"].items()
    )
    expert = shortest_derivation(initial, rules, target)
    if expert is None or not expert:
        raise ValueError("question has no non-empty executable derivation")

    # Build the global, outcome-free action catalogue from every application
    # encountered while saturating the monotonic theory.
    closure = initial
    catalogue: dict[str, RuleApplication] = {}
    while True:
        available = rule_applications(closure, rules)
        if not available:
            break
        catalogue.update((application.text, application) for application in available)
        closure = closure | {application.conclusion for application in available}
    catalogue_text = tuple(sorted(catalogue))

    transitions = []
    state = initial
    for executed in expert:
        available = list(rule_applications(state, rules))
        by_text = {application.text: application for application in available}
        if executed.text not in by_text:
            raise AssertionError("expert application is not available")
        alternatives = [
            application for application in available
            if application.text != executed.text
        ]
        if counterfactual_k is not None:
            alternatives = alternatives[:max(0, int(counterfactual_k))]
        counterfactuals = []
        for alternative in alternatives:
            next_state = state | {alternative.conclusion}
            continuation = shortest_derivation(next_state, rules, target)
            rollout = tuple(
                render_fact(application.conclusion)
                for application in (continuation or ())[
                    :max(teacher_horizon - 1, 0)
                ]
            )
            counterfactuals.append(Counterfactual(
                alternative.text,
                render_fact(alternative.conclusion),
                (rollout,),
            ))
        state = state | {executed.conclusion}
        transitions.append(ObservedTransition(
            executed.text,
            render_fact(executed.conclusion),
            catalogue_text,
            tuple(sorted(by_text)),
            tuple(counterfactuals),
        ))
    if target not in state:
        raise AssertionError("compiled ProofWriter derivation misses its query")
    prompt = tuple(
        [str(item["text"]) for item in record["triples"].values()]
        + [str(item["text"]) for item in record["rules"].values()]
    )
    return ObservedActionEpisode(
        episode_id=f"proofwriter-{record['id']}-{question_id}",
        domain="proofwriter",
        split=split,
        prompt=prompt,
        goal=f"prove: {render_fact(target)}",
        transitions=tuple(transitions),
        metadata={
            "query_depth": int(question.get("QDep", len(expert))),
            "optimal_derivation_length": len(expert),
            "symbolic_environment": True,
            "environment_spec": {
                "initial": [list(fact) for fact in sorted(initial)],
                "target": list(target),
                "rules": [
                    {
                        "rule_id": rule.rule_id,
                        "text": rule.text,
                        "antecedents": [list(fact) for fact in rule.antecedents],
                        "conclusion": list(rule.conclusion),
                    }
                    for rule in rules
                ],
            },
        },
    )
