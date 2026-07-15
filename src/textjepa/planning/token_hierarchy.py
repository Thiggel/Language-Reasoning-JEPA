"""Model-specific helpers for recursive token-hierarchy planning."""

from __future__ import annotations

import torch


def macro_codes(model, token_ids: torch.Tensor, through_level: int | None = None):
    """Encode complete nested token chunks without requiring higher chunks.

    ``through_level`` prevents a partial lower-level history from needlessly
    constructing empty higher levels during receding-horizon execution.
    """
    source = model.token_action(token_ids)
    source_stride, codes = 1, []
    for level_index, (span, level) in enumerate(zip(model.level_spans, model.levels)):
        ratio = span // source_stride
        count = source.shape[1] // ratio
        if count == 0:
            source = source.new_zeros(token_ids.shape[0], 0, level.action.d_macro)
        else:
            source = level.action(
                source[:, :count * ratio].reshape(-1, ratio, source.shape[-1])
            ).reshape(token_ids.shape[0], count, level.action.d_macro)
        codes.append(source)
        if through_level is not None and level_index >= through_level:
            break
        source_stride = span
    return codes


def feedback_levels_to_invalidate(
    mode: str,
    lower_drift: float | None,
    threshold: float,
    n_levels: int,
) -> tuple[int, ...]:
    """Upper cache levels invalidated after one bottom macro executes."""
    if mode == "boundary":
        return ()
    if mode == "l1_feedback":
        return tuple(range(1, n_levels))
    if mode == "adaptive":
        if lower_drift is not None and lower_drift > threshold:
            return tuple(range(1, n_levels))
        return ()
    raise ValueError(f"unknown feedback mode: {mode}")


def remaining_to_boundary(position: int, span: int) -> int:
    """Number of primitive steps remaining in the current fixed span."""
    if span < 1 or position < 0:
        raise ValueError("position must be nonnegative and span positive")
    remainder = position % span
    return span if remainder == 0 else span - remainder
