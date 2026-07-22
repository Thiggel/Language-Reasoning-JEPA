"""ALFWorld trace compiler for the common observed-action contract.

Collection is intentionally separated from training.  A collector must store
both a non-oracle grounded command catalogue and ALFWorld's privileged
``admissible_commands`` labels.  The compiler refuses traces whose expert
action is absent from either set, preventing a silent fallback to the oracle
menu at deployment.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import re
import traceback

from textjepa.data.observed_action import (
    Counterfactual,
    ObservedActionEpisode,
    ObservedTransition,
)


CATALOGUE_POLICY_VERSION = "observed-entities-v1"


def _clean_entity(value: str) -> str:
    value = value.strip().lower().strip(" .;:\n\r\t")
    value = re.sub(r"^(?:a|an|the)\s+", "", value)
    return re.sub(r"\s+", " ", value)


def observed_entities(description: str) -> tuple[str, ...]:
    """Extract numbered entities only from text the agent has observed.

    ALFWorld descriptions enumerate entities after ``you see``.  This parser
    deliberately does not inspect TextWorld facts or admissible commands.
    """
    entities = []
    for clause in re.split(r"\byou see\b", description, flags=re.IGNORECASE)[1:]:
        clause = re.split(
            r"\byour task is to\b", clause, maxsplit=1, flags=re.IGNORECASE
        )[0]
        clause = re.sub(r"\band\b", ",", clause, flags=re.IGNORECASE)
        for part in clause.split(","):
            entity = _clean_entity(part)
            if entity and entity != "nothing" and entity not in entities:
                entities.append(entity)
    return tuple(entities)


def observed_action_catalogue(
    initial_observation: str,
    observation_history: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Ground the fixed ALFWorld grammar from observation history alone.

    Initial entities are navigable receptacles.  Later observations reveal
    manipulable objects.  The cross product intentionally contains impossible
    actions; learning feasibility is part of the experiment.  The catalogue
    grows when an object is observed, but never uses the oracle legal menu.
    """
    receptacles = list(observed_entities(initial_observation))
    observed = []
    for description in observation_history:
        for entity in observed_entities(description):
            if entity not in observed:
                observed.append(entity)
    # Objects can themselves be tools or lamps.  Keep receptacles in the
    # entity pool because ALFWorld occasionally describes them again locally.
    objects = [value for value in observed if value not in receptacles]
    lamps = [value for value in observed if "lamp" in value]
    microwaves = [value for value in receptacles if "microwave" in value]
    fridges = [value for value in receptacles if "fridge" in value]
    cleaners = [
        value for value in receptacles
        if "sink" in value or "bathtub" in value
    ]
    knives = [value for value in (*observed, *receptacles) if "knife" in value]
    commands = ["inventory", "look", "help"]
    for receptacle in receptacles:
        commands.extend((
            f"go to {receptacle}", f"open {receptacle}",
            f"close {receptacle}", f"examine {receptacle}",
        ))
    for obj in objects:
        commands.append(f"examine {obj}")
        for receptacle in receptacles:
            commands.extend((
                f"take {obj} from {receptacle}",
                f"move {obj} to {receptacle}",
            ))
        commands.extend(
            f"heat {obj} with {value}" for value in microwaves
        )
        commands.extend(f"cool {obj} with {value}" for value in fridges)
        commands.extend(f"clean {obj} with {value}" for value in cleaners)
        commands.extend(f"slice {obj} with {value}" for value in knives)
    commands.extend(f"use {value}" for value in lamps)
    return tuple(dict.fromkeys(commands))


def split_task_observation(value: str) -> tuple[str, str]:
    marker = "Your task is to: "
    if marker not in value:
        raise ValueError("ALFWorld reset observation has no task description")
    observation, _, task = value.partition(marker)
    return observation.strip(), task.strip()


class AlfworldTextSession:
    """One optional-dependency TextWorld game session.

    The import lives here so training compiled JSONL does not require
    ALFWorld.  Expert information is requested only during collection.
    """

    def __init__(
        self, gamefile: str | Path, with_expert: bool, max_steps: int = 200,
    ):
        try:
            import textworld
            import textworld.gym
            from alfworld.agents.environment.alfred_tw_env import (
                AlfredDemangler,
                AlfredExpert,
                AlfredExpertType,
                AlfredInfos,
            )
        except ImportError as error:
            raise RuntimeError(
                "ALFWorld collection/evaluation requires the text-only "
                "ALFWorld dependency"
            ) from error
        extras = ["gamefile"]
        wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
        if with_expert:
            extras.append("expert_plan")
            wrappers.append(AlfredExpert(
                expert_type=AlfredExpertType.HANDCODED
            ))
        request = textworld.EnvInfos(
            won=True, admissible_commands=True, extras=extras
        )
        env_id = textworld.gym.register_game(
            str(gamefile), request, max_episode_steps=max_steps,
            wrappers=wrappers,
        )
        self.environment = textworld.gym.make(env_id)

    def reset(self):
        return self.environment.reset()

    def step(self, action: str):
        return self.environment.step(action)

    def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if close is not None:
            close()


def _interactive_session_worker(connection, gamefile: str, max_steps: int) -> None:
    """Own one TextWorld engine in a short-lived process.

    Fast Downward leaves its private shared-library mapping alive for the
    lifetime of the importing process.  Interactive paper evaluation spans
    hundreds of episodes, so each episode must own and release that mapping
    in a disposable worker.
    """
    session = None
    try:
        session = AlfworldTextSession(gamefile, with_expert=False,
                                      max_steps=max_steps)
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "reset":
                result = session.reset()
            elif command == "step":
                result = session.step(str(payload))
            elif command == "close":
                connection.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown ALFWorld worker command: {command}")
            connection.send(("ok", result))
    except EOFError:
        return
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if session is not None:
            session.close()
        connection.close()


class AlfworldInteractiveSession:
    """Process-isolated TextWorld session used by closed-loop evaluation."""

    def __init__(self, gamefile: str | Path, max_steps: int = 200):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(
            target=_interactive_session_worker,
            args=(child, str(gamefile), int(max_steps)),
            daemon=True,
        )
        self.process.start()
        child.close()
        status, payload = self.connection.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(f"ALFWorld worker failed to start:\n{payload}")

    def _request(self, command: str, payload=None):
        if not self.process.is_alive():
            raise RuntimeError("ALFWorld worker exited unexpectedly")
        self.connection.send((command, payload))
        status, result = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"ALFWorld worker failed:\n{result}")
        return result

    def reset(self):
        return self._request("reset")

    def step(self, action: str):
        return self._request("step", action)

    def close(self) -> None:
        process = getattr(self, "process", None)
        connection = getattr(self, "connection", None)
        if process is None:
            return
        if process.is_alive():
            try:
                connection.send(("close", None))
                connection.recv()
            except (BrokenPipeError, EOFError):
                pass
            process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        if connection is not None:
            connection.close()
        self.process = None
        self.connection = None


def _won(info: dict) -> bool:
    return bool(info.get("won", False))


def _expert(info: dict) -> str:
    plan = info.get("extra.expert_plan", ())
    if not plan:
        raise RuntimeError("ALFWorld expert returned no action")
    return str(plan[0])


def _branch_counterfactual(
    session: AlfworldTextSession,
    prefix: list[str],
    alternative: str,
    horizon: int,
    max_steps: int,
) -> dict | None:
    try:
        _, info = session.reset()
        for action in prefix:
            _, _, done, info = session.step(action)
            if done:
                return None
        observation, _, done, info = session.step(alternative)
        immediate = str(observation)
        continuation = []
        for _ in range(max_steps - len(prefix) - 1):
            if done:
                break
            action = _expert(info)
            observation, _, done, info = session.step(action)
            if len(continuation) < horizon:
                continuation.append(str(observation))
        if not _won(info):
            return None
        return {
            "action": alternative,
            "outcome": immediate,
            "teacher_rollouts": [continuation],
        }
    except Exception:
        return None


def collect_alfworld_record(
    gamefile: str | Path,
    data_root: str | Path,
    split: str,
    seed: int,
    counterfactual_k: int = 0,
    teacher_horizon: int = 8,
    max_steps: int = 200,
) -> dict:
    """Execute the official hand-coded expert and capture replayable labels."""
    gamefile, data_root = Path(gamefile).resolve(), Path(data_root).resolve()
    session = AlfworldTextSession(gamefile, with_expert=True, max_steps=max_steps)
    actions, observations, steps = [], [], []
    try:
        reset_observation, info = session.reset()
        initial, task = split_task_observation(str(reset_observation))
        observations.append(initial)
        done = False
        for step_index in range(max_steps):
            action = _expert(info)
            catalogue = observed_action_catalogue(initial, observations)
            available = tuple(str(value) for value in info["admissible_commands"])
            if action not in catalogue:
                raise RuntimeError(
                    f"non-oracle catalogue misses expert at step {step_index}: "
                    f"{action!r}"
                )
            alternatives = [
                value for value in available
                if value != action and value in catalogue
            ]
            rng = random.Random(f"{seed}:{gamefile}:{step_index}")
            rng.shuffle(alternatives)
            counterfactuals = []
            for alternative in alternatives:
                if len(counterfactuals) >= counterfactual_k:
                    break
                branch = _branch_counterfactual(
                    session, actions, alternative, teacher_horizon, max_steps
                )
                if branch is not None:
                    counterfactuals.append(branch)
            # Counterfactual branches reset and mutate the shared engine.
            # Restore the factual prefix before taking the expert action.
            if alternatives and counterfactual_k > 0:
                _, info = session.reset()
                for previous_action in actions:
                    _, _, done, info = session.step(previous_action)
                    if done:
                        raise RuntimeError(
                            "factual prefix terminated while restoring branch"
                        )
            next_observation, _, done, next_info = session.step(action)
            steps.append({
                "catalogue": list(catalogue),
                "admissible_commands": list(available),
                "expert_action": action,
                "next_observation": str(next_observation),
                "counterfactuals": counterfactuals,
            })
            actions.append(action)
            observations.append(str(next_observation))
            info = next_info
            if done:
                break
        if not done or not _won(info):
            raise RuntimeError("expert did not win within the collection budget")
    finally:
        session.close()
    relative = str(gamefile.relative_to(data_root))
    episode_id = hashlib.sha256(relative.encode()).hexdigest()[:20]
    return {
        "schema_version": 1,
        "episode_id": episode_id,
        "split": split,
        "gamefile_relative": relative,
        "initial_observation": initial,
        "task": task,
        "task_type": gamefile.parent.parent.name,
        "won": True,
        "catalogue_policy": CATALOGUE_POLICY_VERSION,
        "steps": steps,
    }


def replay_alfworld_record(record: dict, data_root: str | Path) -> None:
    """Fail unless the stored expert trajectory deterministically replays."""
    gamefile = Path(data_root) / record["gamefile_relative"]
    session = AlfworldTextSession(gamefile, with_expert=False)
    try:
        reset_observation, info = session.reset()
        initial, task = split_task_observation(str(reset_observation))
        if initial != record["initial_observation"] or task != record["task"]:
            raise RuntimeError("reset observation/task changed during replay")
        for index, step in enumerate(record["steps"]):
            available = tuple(str(value) for value in info["admissible_commands"])
            if set(available) != set(step["admissible_commands"]):
                raise RuntimeError(f"admissible commands changed at step {index}")
            observation, _, done, info = session.step(step["expert_action"])
            if str(observation) != step["next_observation"]:
                raise RuntimeError(f"observation changed at step {index}")
        if not done or not _won(info):
            raise RuntimeError("stored trajectory does not win during replay")
    finally:
        session.close()


def compile_alfworld_trace(record: dict, split: str) -> ObservedActionEpisode:
    if record.get("catalogue_policy") not in {None, CATALOGUE_POLICY_VERSION}:
        raise ValueError("unknown ALFWorld catalogue policy")
    transitions = []
    for index, step in enumerate(record["steps"]):
        catalogue = tuple(str(value) for value in step["catalogue"])
        available = tuple(str(value) for value in step["admissible_commands"])
        action = str(step["expert_action"])
        if action not in catalogue:
            raise ValueError(
                f"step {index}: non-oracle catalogue misses expert action"
            )
        if action not in available:
            raise ValueError(
                f"step {index}: expert action is not ALFWorld-admissible"
            )
        counterfactuals = tuple(
            Counterfactual.from_dict(value)
            for value in step.get("counterfactuals", [])
        )
        transitions.append(ObservedTransition(
            action=action,
            outcome=str(step["next_observation"]),
            catalogue=catalogue,
            available=available,
            counterfactuals=counterfactuals,
        ))
    if not record.get("won", False):
        raise ValueError("ALFWorld expert trace does not satisfy the task")
    return ObservedActionEpisode(
        episode_id=f"alfworld-{record['episode_id']}",
        domain="alfworld-textworld",
        split=split,
        prompt=(str(record["initial_observation"]), str(record["task"])),
        goal=str(record["task"]),
        transitions=tuple(transitions),
        metadata={
            "task_type": record.get("task_type", "unknown"),
            "expert_length": len(transitions),
            "oracle_availability_labels": True,
            "gamefile_relative": record.get("gamefile_relative"),
            "catalogue_policy": record.get(
                "catalogue_policy", "legacy-explicit"
            ),
            "raw_trace_sha256": hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()
            ).hexdigest(),
        },
    )
