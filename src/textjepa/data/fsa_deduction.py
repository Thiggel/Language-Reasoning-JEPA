"""Length-controlled synthetic logical-deduction domain (FSA proof problems).

Provenance
----------
Ported from the project owner's own ``synthetic-RLVL`` repository on Alex
(``~/synthetic-RLVL/synthetic_dataset.py``, generator
``LogicDatasetGenerator._generate_hard_fsa_core``, difficulty ``hard_fsa``),
described in ``docs/hfsa_depth_scaling_plan_2026-05-19.md``.  Both repositories
are the owner's unpublished research code, so no third-party licence applies;
this file is an independent re-implementation of the sampling recipe, not a
copy of the training stack.

What the source generator does
------------------------------
Each problem is a finite-state automaton unrolled over ``depth`` steps:

* constants ``c0 .. cD`` are the positions of the trajectory,
* ``c0`` is given a *state* word and a *marker* word,
* at every layer there are ``K`` locally plausible branch transitions of the
  form ``marker(c_t) & state(c_t) -> state'(c_{t+1})`` followed by
  ``state'(c_{t+1}) -> marker'(c_{t+1})``,
* exactly one branch is derivable from the currently derived marker/state
  pair, the other ``K-1`` remain textually coherent,
* the gold derivation therefore has ``2D - 1`` inference steps (``D`` state
  atoms and ``D-1`` marker atoms), and skipping or mis-selecting one step
  changes every later marker and state.

``depth`` is the single length knob and is unbounded in practice: state words
are reused at distinct constants, so long trajectories stay well-formed.

How it is expressed here
------------------------
The problem is emitted as a ground Datalog theory reusing the ProofWriter
fact/rule structures, so the compiled episodes share ProofWriter's action
phrasing (``apply rule: ... using facts: ...``), its executor, and the
domain-neutral observed-action schema.  The symbolic theory is an environment,
never a model feature.

Per-step candidate catalogue
----------------------------
The full ground theory has ``2 * K * D`` rules, which is too large to write out
once per transition.  The compiled per-step catalogue is therefore bounded and
defined purely from observable state: it always contains every currently
applicable rule application, then fills up with the rules whose destination
constant lies closest to the derived frontier, ties broken by a hash of the
episode id and step index.  Nothing about the gold path enters that ordering.
The episode-level catalogue seen by the closed-loop evaluator is the union over
steps, which recovers essentially the whole ground theory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

from textjepa.data.observed_action import (
    Counterfactual,
    ObservedActionEpisode,
    ObservedTransition,
)
from textjepa.data.proofwriter import (
    Fact,
    ProofRule,
    RuleApplication,
    render_fact,
)

# Verbatim from synthetic-RLVL ``HARD_V5_STATE_WORDS``.
STATE_WORDS: tuple[str, ...] = (
    "amber", "cobalt", "ivory", "olive", "ruby", "slate", "coral", "lime",
    "pearl", "teal", "maple", "cedar", "hazel", "birch", "juniper", "willow",
    "laurel", "orchid", "violet", "poppy", "elm", "granite", "harbor", "meadow",
)
MARKER_WORDS: tuple[str, ...] = (
    "north", "south", "east", "west", "open", "closed", "bright", "dim",
)
# The source generator draws state predicates from a 26-symbol pool shared with
# the markers; keeping the same bound preserves the sampling distribution.
MAX_STATE_SYMBOLS = 26
INVALID_OUTCOME = "The proposed inference is invalid and no fact is added ."


@dataclass(frozen=True)
class FsaProblem:
    problem_id: str
    depth: int
    branching_factor: int
    initial: frozenset[Fact]
    rules: tuple[ProofRule, ...]
    target: Fact
    answer: str
    path_states: tuple[str, ...]
    path_markers: tuple[str, ...]

    @property
    def constants(self) -> tuple[str, ...]:
        return tuple(f"c{index}" for index in range(self.depth + 1))


def _atom(constant: str, word: str) -> Fact:
    return (constant, "is", word, "+")


def _rule_text(antecedents: tuple[Fact, ...], conclusion: Fact) -> str:
    clauses = " and ".join(
        f"{fact[0]} is {fact[2]}" for fact in antecedents
    )
    return f"If {clauses}, then {conclusion[0]} is {conclusion[2]}."


def problem_rng(seed: int, index: int) -> random.Random:
    """Reproduce the source generator's per-index deterministic stream."""
    digest = hashlib.sha256(f"{seed}|{index}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def sample_fsa_problem(
    seed: int,
    index: int,
    depth: int,
    branching_factor: int = 4,
    side_facts_per_step: int = 0,
    side_kind: str = "side",
    dead_end_length: int = 1,
) -> FsaProblem:
    """Sample one FSA deduction problem with ``depth`` gold inference layers.

    ``side_facts_per_step`` is an optional deviation from the source recipe: it
    adds that many applicable-but-off-path rules per layer so the feasible
    menu is larger than one.  Zero reproduces the source generator, where
    exactly one rule is applicable at every point.

    ``side_kind`` selects what those off-path rules look like:

    * ``"side"`` (historical): ``If c_k is S, then c_k is <word>-side.`` --
      lexically marked conclusions that no rule consumes, so the necessary
      rule is identifiable from the candidate text alone.
    * ``"dead_end"`` (2026-09): decoy rules with the SAME antecedents and
      surface form as the gold rule (``If c_k is M and c_k is S, then
      c_{k+1} is D.``) whose conclusion starts a decoy chain (marker rule +
      ``dead_end_length`` further layers) that never reaches the target;
      ``dead_end_length = -1`` instead switches onto one of the catalogue's
      full-length off-path branches, which ends at c_depth with a non-target
      state.  The necessary rule can then only be identified by following the
      catalogue forward (lookahead) or backward from the goal.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if branching_factor < 2:
        raise ValueError("branching_factor must be >= 2")
    if side_kind not in {"side", "dead_end"}:
        raise ValueError(f"unknown side_kind: {side_kind}")
    rng = problem_rng(seed, index)
    constants = [f"c{i}" for i in range(depth + 1)]
    max_state_symbols = min(
        len(STATE_WORDS), MAX_STATE_SYMBOLS - branching_factor
    )
    if branching_factor + 1 > max_state_symbols:
        raise ValueError("branching_factor exceeds the available state words")
    num_state_symbols = min(
        max(depth + 1, branching_factor + 1), max_state_symbols
    )
    states = rng.sample(STATE_WORDS, num_state_symbols)
    markers = list(MARKER_WORDS[:branching_factor])

    initial_state = rng.choice(states)
    non_initial = [state for state in states if state != initial_state]
    rng.shuffle(non_initial)
    if len(non_initial) < branching_factor:
        raise ValueError("not enough non-initial states for branch uniqueness")
    initial_markers = markers.copy()
    rng.shuffle(initial_markers)

    for _attempt in range(200):
        branch_states = [[initial_state] for _ in range(branching_factor)]
        branch_markers = [[initial_markers[i]] for i in range(branching_factor)]
        used: set[tuple[str, str]] = set()
        ok = True
        for layer in range(depth):
            destination = constants[layer + 1]
            candidates = [
                state for state in non_initial
                if (state, destination) not in used
            ]
            rng.shuffle(candidates)
            if len(candidates) < branching_factor:
                ok = False
                break
            layer_states = candidates[:branching_factor]
            layer_markers = rng.sample(markers, branching_factor)
            pairs = list(zip(layer_states, layer_markers, strict=True))
            rng.shuffle(pairs)
            for branch, (state, marker) in enumerate(pairs):
                branch_states[branch].append(state)
                branch_markers[branch].append(marker)
                used.add((state, destination))
        if ok:
            break
    else:
        raise RuntimeError("failed to build collision-free FSA trajectories")

    initial = frozenset({
        _atom(constants[0], initial_state),
        _atom(constants[0], branch_markers[0][0]),
    })
    rules: list[ProofRule] = []

    def add_rule(antecedents: tuple[Fact, ...], conclusion: Fact) -> None:
        rules.append(ProofRule(
            rule_id=f"r{len(rules) + 1}",
            text=_rule_text(antecedents, conclusion),
            antecedents=antecedents,
            conclusion=conclusion,
        ))

    side_words = [
        word for word in STATE_WORDS if word not in states
    ] or list(STATE_WORDS)
    # (state word, constant) pairs already claimed by the branches; decoy
    # states must not collide with them (a collision would silently splice a
    # decoy into a real branch).
    claimed: set[tuple[str, str]] = set(used) | {
        (branch_states[b][0], constants[0]) for b in range(branching_factor)
    }
    decoy_rng = random.Random(rng.random()) if side_kind == "dead_end" else None

    def fresh_state(constant: str) -> str:
        pool = [w for w in STATE_WORDS if (w, constant) not in claimed]
        if not pool:
            raise RuntimeError("no free state word for a decoy fact")
        word = decoy_rng.choice(pool)
        claimed.add((word, constant))
        return word

    for step in range(depth):
        source, destination = constants[step], constants[step + 1]
        order = list(range(branching_factor))
        rng.shuffle(order)
        for branch in order:
            antecedents = (
                _atom(source, branch_markers[branch][step]),
                _atom(source, branch_states[branch][step]),
            )
            add_rule(antecedents, _atom(destination, branch_states[branch][step + 1]))
            add_rule(
                (_atom(destination, branch_states[branch][step + 1]),),
                _atom(destination, branch_markers[branch][step + 1]),
            )
        for extra in range(side_facts_per_step):
            if side_kind == "side":
                word = f"{side_words[(step + extra) % len(side_words)]}-side"
                add_rule(
                    (_atom(source, branch_states[0][step]),),
                    _atom(source, word),
                )
                continue
            gold_antecedents = (
                _atom(source, branch_markers[0][step]),
                _atom(source, branch_states[0][step]),
            )
            if dead_end_length < 0:
                # switch onto a full-length off-path branch
                other = 1 + (extra % (branching_factor - 1))
                add_rule(
                    gold_antecedents,
                    _atom(destination, branch_states[other][step + 1]),
                )
                continue
            # fresh decoy chain: switch rule, marker rule, then
            # ``dead_end_length`` further layers, each with its own marker.
            antecedents = gold_antecedents
            for hop in range(dead_end_length + 1):
                layer = step + 1 + hop
                if layer > depth:
                    break
                here = constants[layer]
                d_state = fresh_state(here)
                d_marker = decoy_rng.choice(markers)
                add_rule(antecedents, _atom(here, d_state))
                add_rule((_atom(here, d_state),), _atom(here, d_marker))
                antecedents = (_atom(here, d_marker), _atom(here, d_state))

    if side_kind == "dead_end":
        # catalogue order must carry no information about which feasible rule
        # is the gold one (the historical generator lists a layer's branch
        # rules before its decoys); re-number after shuffling.
        decoy_rng.shuffle(rules)
        rules = [
            ProofRule(rule_id=f"r{i + 1}", text=r.text,
                      antecedents=r.antecedents, conclusion=r.conclusion)
            for i, r in enumerate(rules)
        ]
    target = _atom(constants[depth], branch_states[0][depth])
    tag = "" if side_kind == "side" else f"-de{dead_end_length}"
    return FsaProblem(
        problem_id=f"s{seed}-d{depth}-k{branching_factor}-i{index}{tag}",
        depth=depth,
        branching_factor=branching_factor,
        initial=initial,
        rules=tuple(rules),
        target=target,
        answer=branch_states[0][depth],
        path_states=tuple(branch_states[0]),
        path_markers=tuple(branch_markers[0]),
    )


def canonical_identity(problem: FsaProblem) -> str:
    """Content identity of a problem, independent of its sampling index."""
    payload = "\n".join(
        [f"depth={problem.depth}", f"k={problem.branching_factor}"]
        + sorted(render_fact(fact) for fact in problem.initial)
        + sorted(rule.text for rule in problem.rules)
        + [render_fact(problem.target)]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _ground_application(rule: ProofRule) -> RuleApplication:
    return RuleApplication(rule, rule.antecedents, rule.conclusion)


def ground_applications(
    state: frozenset[Fact], rules: tuple[ProofRule, ...],
) -> tuple[RuleApplication, ...]:
    """Applicable, non-redundant applications of a fully ground theory.

    Equivalent to :func:`textjepa.data.proofwriter.rule_applications` for
    variable-free rules, but avoids the generic matcher's quadratic scan over
    facts, which dominates compilation at depth 50.
    """
    return tuple(
        _ground_application(rule) for rule in rules
        if rule.conclusion not in state
        and all(fact in state for fact in rule.antecedents)
    )


def _layer_index(fact: Fact) -> int:
    return int(fact[0][1:])


def expert_derivation(problem: FsaProblem) -> tuple[RuleApplication, ...]:
    """Forward-chain to the target along the gold path.

    Applies, per layer, the rule concluding the next path state and then the
    rule concluding its marker (2*depth - 1 applications).  In the historical
    ``side`` generator this is exactly the greedy chain; with decoy rules the
    greedy chain would wander, so the path is followed explicitly.
    """
    if problem.path_states and problem.path_markers:
        return _path_derivation(problem)
    state = problem.initial
    derivation: list[RuleApplication] = []
    for _ in range(4 * problem.depth + 4):
        if problem.target in state:
            return tuple(derivation)
        available = ground_applications(state, problem.rules)
        if not available:
            break
        chosen = None
        for application in available:
            if _layer_index(application.conclusion) >= max(
                (_layer_index(fact) for fact in state), default=0
            ):
                chosen = application
                break
        chosen = chosen or available[0]
        derivation.append(chosen)
        state = state | {chosen.conclusion}
    raise RuntimeError(f"FSA problem {problem.problem_id} did not reach its target")


def _path_derivation(problem: FsaProblem) -> tuple[RuleApplication, ...]:
    constants = problem.constants
    state = set(problem.initial)
    by_conclusion: dict[Fact, list[ProofRule]] = {}
    for rule in problem.rules:
        by_conclusion.setdefault(rule.conclusion, []).append(rule)
    derivation: list[RuleApplication] = []

    def apply(conclusion: Fact) -> None:
        for rule in by_conclusion.get(conclusion, []):
            if all(fact in state for fact in rule.antecedents):
                derivation.append(_ground_application(rule))
                state.add(conclusion)
                return
        raise RuntimeError(
            f"FSA problem {problem.problem_id}: no applicable rule concludes "
            f"{conclusion}"
        )

    for layer in range(1, problem.depth + 1):
        apply(_atom(constants[layer], problem.path_states[layer]))
        if problem.target in state:
            return tuple(derivation)
        apply(_atom(constants[layer], problem.path_markers[layer]))
    raise RuntimeError(f"FSA problem {problem.problem_id} did not reach its target")


def _bounded_catalogue(
    problem: FsaProblem,
    state: frozenset[Fact],
    available: tuple[RuleApplication, ...],
    episode_id: str,
    step: int,
    catalogue_cap: int,
) -> tuple[RuleApplication, ...]:
    frontier = max((_layer_index(fact) for fact in state), default=0)
    selected = {application.text: application for application in available}
    pool = [
        _ground_application(rule) for rule in problem.rules
        if _rule_key(rule) not in selected
    ]

    def order_key(application: RuleApplication) -> tuple:
        distance = abs(_layer_index(application.conclusion) - frontier)
        tie = hashlib.sha256(
            f"{episode_id}|{step}|{application.text}".encode()
        ).hexdigest()
        return (distance, tie)

    for application in sorted(pool, key=order_key):
        if len(selected) >= catalogue_cap:
            break
        selected.setdefault(application.text, application)
    return tuple(selected[key] for key in sorted(selected))


def _rule_key(rule: ProofRule) -> str:
    return _ground_application(rule).text


def compile_fsa_episode(
    problem: FsaProblem,
    split: str,
    teacher_horizon: int = 4,
    counterfactual_k: int = 3,
    catalogue_cap: int = 12,
) -> ObservedActionEpisode:
    expert = expert_derivation(problem)
    if not expert:
        raise ValueError("FSA problem has an empty derivation")
    episode_id = f"fsa-deduction-{problem.problem_id}"
    outcomes = [render_fact(application.conclusion) for application in expert]
    actions = [application.text for application in expert]

    transitions = []
    state = problem.initial
    for step, executed in enumerate(expert):
        available = ground_applications(state, problem.rules)
        by_text = {application.text: application for application in available}
        if executed.text not in by_text:
            raise AssertionError("expert application is not available")
        catalogue = _bounded_catalogue(
            problem, state, available, episode_id, step, catalogue_cap,
        )
        catalogue_text = tuple(application.text for application in catalogue)
        # Teacher continuation from an unchanged state: every recorded
        # alternative is either invalid or redundant, so the remaining gold
        # suffix is the continuation in both cases.
        rollout_states = tuple(outcomes[step:][:max(teacher_horizon - 1, 0)])
        rollout_actions = tuple(actions[step:][:max(teacher_horizon - 1, 0)])
        counterfactuals = []
        for application in catalogue:
            if application.text == executed.text:
                continue
            if len(counterfactuals) >= counterfactual_k:
                break
            if application.text in by_text:
                outcome = render_fact(application.conclusion)
            elif all(fact in state for fact in application.premises):
                # Valid but redundant: the conclusion is already derived.
                outcome = render_fact(application.conclusion)
            else:
                outcome = INVALID_OUTCOME
            counterfactuals.append(Counterfactual(
                application.text, outcome,
                (rollout_states,), (rollout_actions,),
            ))
        transitions.append(ObservedTransition(
            executed.text,
            render_fact(executed.conclusion),
            catalogue_text,
            tuple(sorted(by_text)),
            tuple(counterfactuals),
        ))
        state = state | {executed.conclusion}
    if problem.target not in state:
        raise AssertionError("compiled FSA derivation misses its target")

    prompt = tuple(
        [render_fact(fact) for fact in sorted(problem.initial)]
        + [rule.text for rule in problem.rules]
    )
    return ObservedActionEpisode(
        episode_id=episode_id,
        domain="fsa-deduction",
        split=split,
        prompt=prompt,
        goal=f"prove: {render_fact(problem.target)}",
        transitions=tuple(transitions),
        metadata={
            "depth": problem.depth,
            "branching_factor": problem.branching_factor,
            "optimal_derivation_length": len(expert),
            "answer": problem.answer,
            "identity": canonical_identity(problem),
            "symbolic_environment": True,
            "environment_spec": {
                "initial": [list(fact) for fact in sorted(problem.initial)],
                "target": list(problem.target),
                "rules": [
                    {
                        "rule_id": rule.rule_id,
                        "text": rule.text,
                        "antecedents": [list(fact) for fact in rule.antecedents],
                        "conclusion": list(rule.conclusion),
                    }
                    for rule in problem.rules
                ],
            },
        },
    )
