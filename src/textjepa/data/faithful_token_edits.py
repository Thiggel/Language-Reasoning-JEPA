"""Token-edit trajectories over official iGSM solution text.

Corruption and repair operate only on rendered tokens.  No dependency graph,
remaining-work label, feasibility signal, or symbolic action ordering enters
the edit trajectory.
"""

from __future__ import annotations

import random

from torch.utils.data import Dataset

from textjepa.data.edits.dataset import collate_edits
from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab
from textjepa.data.vocab import Vocab


TOKEN_EDIT_WORDS = ["token", "position", "with"]
MASK_TOKEN = "<mask>"
OPS = {"delete": 0, "insert": 1, "replace": 2}


def faithful_token_edit_vocab(max_position: int = 1024) -> Vocab:
    base = cached_faithful_vocab()
    words = [
        token for token in base.token_to_id
        if token not in {base.PAD, base.UNK}
    ]
    words.extend(TOKEN_EDIT_WORDS + [MASK_TOKEN])
    words.extend(str(index) for index in range(max_position + 1))
    return Vocab(words)


def _flat_length(buffer: list[list[int]]) -> int:
    return sum(len(sentence) for sentence in buffer)


def _position(
    buffer: list[list[int]], position: int
) -> tuple[int, int]:
    """Map a flattened token position to its official step and offset.

    A position exactly on a step boundary belongs to the step on its right.
    Corrupting deletions therefore avoid the final token of a step: otherwise
    its inverse insertion would be ambiguous from the literal token position
    alone.  Insertions at boundaries and deletions of the first token of a
    step remain fully causal and invertible.
    """
    if position < 0 or position >= _flat_length(buffer):
        raise IndexError(position)
    start = 0
    for sentence_index, sentence in enumerate(buffer):
        end = start + len(sentence)
        if position < end:
            return sentence_index, position - start
        start = end
    raise AssertionError("unreachable flattened token position")


def _apply(
    buffer: list[list[int]], action: tuple[str, int, int | None]
) -> None:
    """Apply one literal edit while retaining official step boundaries."""
    kind, position, token = action
    if kind == "delete":
        sentence, offset = _position(buffer, position)
        if len(buffer[sentence]) <= 1:
            raise AssertionError("token edit unexpectedly emptied a step")
        buffer[sentence].pop(offset)
    elif kind == "insert":
        if position == _flat_length(buffer):
            sentence, offset = len(buffer) - 1, len(buffer[-1])
        else:
            sentence, offset = _position(buffer, position)
        buffer[sentence].insert(offset, int(token))
    elif kind == "replace":
        sentence, offset = _position(buffer, position)
        buffer[sentence][offset] = int(token)
    else:
        raise ValueError(kind)


def _render_action(
    vocab: Vocab, action: tuple[str, int, int | None]
) -> list[int]:
    kind, position, token = action
    rendered = f"{kind} token position {position}"
    if token is not None:
        rendered += f" with {vocab.id_to_token[int(token)]}"
    return vocab.encode(rendered + " .")


def _counterfactual_action(
    buffer: list[list[int]], rng: random.Random, source: str, candidate: int,
    operation_order: tuple[str, ...] | None = None,
) -> tuple[str, int, int | None]:
    """Sample an unlabeled, mechanically executable edit from observed text."""
    token_pool = list(dict.fromkeys(
        token for sentence in buffer for token in sentence
    ))
    replaceable = []
    deletable = []
    start = 0
    for sentence in buffer:
        replaceable.extend(range(start, start + len(sentence)))
        if len(sentence) > 1:
            deletable.extend(range(start, start + len(sentence)))
        start += len(sentence)
    valid = ["insert"]
    if len(token_pool) > 1 and replaceable:
        valid.append("replace")
    if deletable:
        valid.append("delete")
    if source == "uniform_local":
        kind = rng.choice(valid)
    elif source == "mixed":
        # A stable operation-balanced prefix makes small-K versus large-K
        # comparisons vary coverage rather than the expert trajectory.
        preferred = ("replace", "insert", "delete")[candidate % 3]
        kind = preferred if preferred in valid else rng.choice(valid)
    elif source == "deployable_mixed":
        if not operation_order:
            raise ValueError("deployable_mixed requires an operation order")
        preferred = operation_order[candidate % len(operation_order)]
        kind = preferred if preferred in valid else rng.choice(valid)
    else:
        raise ValueError(f"unknown counterfactual_source: {source}")

    if kind == "insert":
        return kind, rng.randrange(_flat_length(buffer) + 1), rng.choice(token_pool)
    position = rng.choice(deletable if kind == "delete" else replaceable)
    if kind == "delete":
        return kind, position, None
    sentence, offset = _position(buffer, position)
    old = buffer[sentence][offset]
    return kind, position, rng.choice([token for token in token_pool if token != old])


def _counterfactual_exclusions(
    source: str, expert: tuple[str, int, int | None]
) -> set[tuple[str, int, int | None]]:
    """Historical modes exclude the expert; deployable sampling cannot."""
    return set() if source == "deployable_mixed" else {expert}


class FaithfulTokenEditDataset(Dataset):
    """Official iGSM prompt plus a generically corrupted solution buffer."""

    def __init__(self, vocab: Vocab, size: int, seed: int, max_op: int = 21,
                 max_edge: int = 28, op_range=(8, 21), min_edits: int = 6,
                 max_edits: int = 16, counterfactual_k: int = 0,
                 counterfactual_source: str = "uniform_local",
                 corruption_mode: str = "mixed", curriculum_epochs: int = 3,
                 fresh_per_epoch: bool = False,
                 **_):
        self.vocab = vocab
        self.seed = seed
        self.min_edits = int(min_edits)
        self.max_edits = int(max_edits)
        self.counterfactual_k = max(0, int(counterfactual_k))
        self.counterfactual_source = str(counterfactual_source)
        self.corruption_mode = str(corruption_mode)
        self.curriculum_epochs = max(1, int(curriculum_epochs))
        self.fresh_per_epoch = bool(fresh_per_epoch)
        self.epoch = 0
        if self.corruption_mode not in {
            "mixed", "mask", "replace", "remove", "curriculum"
        }:
            raise ValueError(f"unknown corruption_mode: {self.corruption_mode}")
        if self.counterfactual_source not in {
            "uniform_local", "mixed", "deployable_mixed"
        }:
            raise ValueError(
                f"unknown counterfactual_source: {self.counterfactual_source}"
            )
        self.source = FaithfulDataset(
            vocab, size=size, seed=seed, max_op=max_op, max_edge=max_edge,
            op_range=tuple(op_range), distractor_prob=0.0,
            max_distractors=0,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def _active_corruption_mode(self) -> str:
        if self.corruption_mode != "curriculum":
            return self.corruption_mode
        progress = self.epoch / max(self.curriculum_epochs - 1, 1)
        if progress < 1 / 3:
            return "mask"
        if progress < 2 / 3:
            return "replace"
        return "mixed"
    def __len__(self):
        return len(self.source)

    def __getitem__(self, index: int) -> dict:
        source = self.source[index]
        prompt = source["prompt"]
        target = [list(sentence) for sentence in source["steps"]]
        mode = self._active_corruption_mode()
        epoch_key = self.epoch if self.fresh_per_epoch else 0
        rng = random.Random(
            f"faithful-token-edit:{self.seed}:{index}:{mode}:{epoch_key}"
        )
        current = [list(sentence) for sentence in target]
        undo: list[tuple[str, int, int | None]] = []
        n_tokens = _flat_length(target)
        upper = min(self.max_edits, max(self.min_edits, n_tokens // 8))
        count = rng.randint(min(self.min_edits, upper), upper)
        token_pool = list(dict.fromkeys(
            token for sentence in target for token in sentence
        ))
        mask_id = self.vocab.token_to_id[MASK_TOKEN]
        for _ in range(count):
            allowed = {
                "mixed": ["replace", "insert"],
                "mask": ["mask"],
                "replace": ["replace"],
                "remove": [],
            }[mode]
            # Deleting a step-final token would erase which side of the
            # structural boundary owns its inverse insertion.  Every other
            # token, including the first token after a boundary, is safe.
            deletable = []
            start = 0
            for sentence in current:
                deletable.extend(range(start, start + len(sentence) - 1))
                start += len(sentence)
            if deletable:
                if mode in {"mixed", "remove"}:
                    allowed.append("delete")
            if not allowed:
                continue
            kind = rng.choice(allowed)
            if kind in {"replace", "mask"}:
                if kind == "mask":
                    candidates = [
                        position for position in range(_flat_length(current))
                        if current[_position(current, position)[0]][
                            _position(current, position)[1]
                        ] != mask_id
                    ]
                    if not candidates:
                        continue
                    position = rng.choice(candidates)
                else:
                    position = rng.randrange(_flat_length(current))
                sentence, offset = _position(current, position)
                old = current[sentence][offset]
                alternatives = (
                    [mask_id] if kind == "mask"
                    else [token for token in token_pool if token != old]
                )
                if not alternatives:
                    continue
                current[sentence][offset] = rng.choice(alternatives)
                undo.append(("replace", position, old))
            elif kind == "delete":
                position = rng.choice(deletable)
                sentence, offset = _position(current, position)
                old = current[sentence].pop(offset)
                undo.append(("insert", position, old))
            else:
                position = rng.randrange(_flat_length(current) + 1)
                _apply(current, ("insert", position, rng.choice(token_pool)))
                undo.append(("delete", position, None))

        repairs = list(reversed(undo))
        buffers = [[list(sentence) for sentence in current]]
        actions, op, positions, content_tokens, remaining, changed = [], [], [], [], [], []
        alt_actions: list[list[list[int]]] = []
        alt_buffers: list[list[list[list[int]]]] = []
        alt_changed: list[list[list[int]]] = []
        alt_ops: list[list[int]] = []
        alt_positions: list[list[int]] = []
        alt_content_tokens: list[list[int]] = []
        for step, action in enumerate(repairs):
            if self.counterfactual_k:
                # Separate keyed streams make alternatives deterministic and
                # prefix-stable without perturbing expert corruption/repair.
                alt_rng = random.Random(
                    f"faithful-token-edit-cf:{self.seed}:{index}:{step}:"
                    f"{self.counterfactual_source}"
                )
                step_actions = []
                step_buffers = []
                step_changed = []
                step_ops = []
                step_positions = []
                step_content_tokens = []
                # Historical sources exclude the expert repair.  The
                # deployment-feasible source must not consult that
                # target-derived action when defining its candidate set.
                sampled = _counterfactual_exclusions(
                    self.counterfactual_source, action
                )
                operation_order = None
                if self.counterfactual_source == "deployable_mixed":
                    base = ("replace", "insert", "delete")
                    offset = (index + step) % len(base)
                    operation_order = base[offset:] + base[:offset]
                attempts = 0
                while len(step_actions) < self.counterfactual_k:
                    candidate = len(step_actions)
                    alternative = _counterfactual_action(
                        current, alt_rng, self.counterfactual_source, candidate,
                        operation_order=operation_order,
                    )
                    attempts += 1
                    if alternative in sampled:
                        if attempts >= 10_000:
                            raise RuntimeError(
                                "could not sample enough distinct edit alternatives"
                            )
                        continue
                    sampled.add(alternative)
                    outcome = [list(sentence) for sentence in current]
                    _apply(outcome, alternative)
                    changed_sentence = next(
                        after for before, after in zip(current, outcome)
                        if before != after
                    )
                    step_actions.append(_render_action(self.vocab, alternative))
                    step_buffers.append(outcome)
                    step_changed.append(list(changed_sentence))
                    alt_kind, alt_position, alt_token = alternative
                    step_ops.append(OPS[alt_kind])
                    step_positions.append(alt_position)
                    step_content_tokens.append(
                        self.vocab.pad_id if alt_token is None else int(alt_token)
                    )
                alt_actions.append(step_actions)
                alt_buffers.append(step_buffers)
                alt_changed.append(step_changed)
                alt_ops.append(step_ops)
                alt_positions.append(step_positions)
                alt_content_tokens.append(step_content_tokens)
            kind, position, token = action
            actions.append(_render_action(self.vocab, action))
            op.append(OPS[kind])
            positions.append(position)
            content_tokens.append(
                self.vocab.pad_id if token is None else int(token)
            )
            _apply(current, action)
            buffers.append([list(sentence) for sentence in current])
            remaining.append(len(repairs) - step - 1)
            before = buffers[-2]
            changed.append(next(
                list(after) for prior, after in zip(before, current)
                if prior != after
            ))
        if current != target:
            raise AssertionError("token-edit undo trajectory did not recover target")
        out = {
            "prompt": prompt,
            "buffers": buffers,
            "actions": actions,
            "op": op,
            "edit_position": positions,
            "edit_content_token": content_tokens,
            "value": [0] * len(actions),
            "remaining": remaining,
            "resolved_n": [_flat_length(b) for b in buffers[1:]],
            "necessary": [1] * len(actions),
            "answer": int(source["answer"]),
            "n_necessary": len(actions),
            "n_vars": 0,
            "index": index,
            "edit_pos": [min(action[1], 15) for action in repairs],
            "changed": changed,
            "defect_masks": [[] for _ in repairs],
        }
        if self.counterfactual_k:
            # Outcomes are deliberately unlabeled: no target-relative
            # remaining-edit, defect, preference, or quality fields.
            out["alt_actions"] = alt_actions
            out["alt_buffers"] = alt_buffers
            out["alt_changed"] = alt_changed
            out["alt_op"] = alt_ops
            out["alt_edit_position"] = alt_positions
            out["alt_edit_content_token"] = alt_content_tokens
        return out


__all__ = [
    "FaithfulTokenEditDataset", "collate_edits", "faithful_token_edit_vocab",
]
