"""ALFWorld trace compiler for the common observed-action contract.

Collection is intentionally separated from training.  A collector must store
both a non-oracle grounded command catalogue and ALFWorld's privileged
``admissible_commands`` labels.  The compiler refuses traces whose expert
action is absent from either set, preventing a silent fallback to the oracle
menu at deployment.
"""

from __future__ import annotations

from textjepa.data.observed_action import (
    Counterfactual,
    ObservedActionEpisode,
    ObservedTransition,
)


def compile_alfworld_trace(record: dict, split: str) -> ObservedActionEpisode:
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
        },
    )
