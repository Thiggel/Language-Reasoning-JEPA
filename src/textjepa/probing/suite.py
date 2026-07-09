"""Probe task registry: what should a good discourse state encode?

Feature sources: state (s_t), pred (teacher-forced F output), rollout
(open-loop F output), delta (s_t - s_{t-1}), action (bottleneck code),
s0 (post-prompt state), final_state.

The value probes on pred/rollout are the headline: the predictor never
sees the step outcome, so decodable values there mean the world model is
doing modular arithmetic in latent space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from textjepa.probing.probes import logistic_probe_accuracy, majority_baseline


@dataclass(frozen=True)
class ProbeTask:
    name: str
    features: str
    labels: str
    description: str


PROBE_TASKS = [
    ProbeTask("value_from_state", "state", "value",
              "value computed at step t, from s_t (encoder saw the outcome)"),
    ProbeTask("value_from_pred", "pred", "value",
              "value at step t from F(s_{t-1}, a_t): latent arithmetic, 1 step"),
    ProbeTask("value_from_rollout", "rollout", "value",
              "value at step t from open-loop rollout: latent arithmetic, t steps"),
    ProbeTask("value_from_chunk_emb", "chunk_emb", "value",
              "value from the online chunk embedding (does the encoder keep it?)"),
    ProbeTask("value_from_chunk_pred", "chunk_pred", "value",
              "value from predicted next-chunk embedding: anchored latent arithmetic"),
    ProbeTask("value_from_chunk_pred_rollout", "chunk_pred_rollout", "value",
              "same, from open-loop rollout states"),
    ProbeTask("value_prev1_from_state", "state", "value_prev1",
              "memory: value established one step earlier, from s_t"),
    ProbeTask("value_prev2_from_state", "state", "value_prev2",
              "memory: value established two steps earlier, from s_t"),
    ProbeTask("op_from_delta", "delta", "op",
              "executed op type from state displacement (Delta-JEPA geometry)"),
    ProbeTask("op_from_action", "action", "op",
              "op type from the bottlenecked action code"),
    ProbeTask("necessary_from_delta", "delta", "necessary",
              "was the step goal-relevant, from displacement"),
    ProbeTask("necessary_from_action", "action", "necessary",
              "goal-relevance from action code alone (context-free control)"),
    ProbeTask("remaining_from_state", "state", "remaining",
              "necessary steps remaining, from s_t (goal geometry)"),
    ProbeTask("resolved_count_from_state", "state", "resolved_n",
              "how many quantities are resolved so far"),
    ProbeTask("answer_from_final", "final_state", "answer",
              "final answer from the terminal state"),
    ProbeTask("answer_from_s0", "s0", "answer",
              "answer from the prompt alone (hard: needs full computation)"),
    ProbeTask("n_necessary_from_s0", "s0", "n_necessary",
              "plan length awareness from the prompt state"),
    ProbeTask("n_vars_from_s0", "s0", "n_vars",
              "problem size from the prompt state"),
]


EDIT_PROBE_TASKS = [
    ProbeTask("stated_answer_from_state", "state", "value",
              "what the buffer currently claims the answer is (incl. absent)"),
    ProbeTask("stated_answer_from_pred", "pred", "value",
              "claimed answer from F(s_{t-1}, a_t): consequence prediction"),
    ProbeTask("editop_from_delta", "delta", "op",
              "edit type (del/ins/repl) from state displacement"),
    ProbeTask("editop_from_action", "action", "op",
              "edit type from the bottlenecked action code"),
    ProbeTask("fixing_from_delta", "delta", "necessary",
              "did the edit reduce defects, from displacement"),
    ProbeTask("defects_from_state", "state", "remaining",
              "defects remaining in the buffer (goal geometry)"),
    ProbeTask("buffer_len_from_state", "state", "resolved_n",
              "current buffer length"),
    ProbeTask("answer_from_final", "final_state", "answer",
              "true answer from the terminal (perfect) buffer state"),
    ProbeTask("initial_defects_from_s0", "s0", "n_necessary",
              "how corrupted is the initial buffer"),
]


def run_probe_suite(
    feats: dict[str, np.ndarray],
    max_n: int = 20000,
    seed: int = 0,
    tasks: list[ProbeTask] | None = None,
) -> pd.DataFrame:
    rows = []
    for task in tasks or PROBE_TASKS:
        if task.features not in feats or task.labels not in feats:
            continue
        x, y = feats[task.features], feats[task.labels]
        valid = y >= 0  # negative labels mark undefined positions
        x, y = x[valid], y[valid]
        if len(x) > max_n:
            idx = np.random.RandomState(seed).permutation(len(x))[:max_n]
            x, y = x[idx], y[idx]
        rows.append(
            {
                "task": task.name,
                "acc": logistic_probe_accuracy(x, y, seed=seed),
                "majority": majority_baseline(y),
                "n": len(y),
                "description": task.description,
            }
        )
    return pd.DataFrame(rows)
