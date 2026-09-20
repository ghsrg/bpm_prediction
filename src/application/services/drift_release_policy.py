from __future__ import annotations


def released_trace_indices(
    current: tuple[int, ...],
    following: tuple[int, ...],
    seen: frozenset[int],
) -> tuple[int, ...]:
    """Return current-window stream indices eligible for the next update."""
    if len(current) != len(set(current)) or len(following) != len(set(following)):
        raise ValueError("Drift windows contain duplicate stream indices.")

    next_indices = set(following)
    return tuple(
        trace_idx
        for trace_idx in current
        if trace_idx not in next_indices and trace_idx not in seen
    )



def eligible_released_trace_indices(
    released: tuple[int, ...],
    *,
    start_trace_index: int,
) -> tuple[int, ...]:
    """Return released stream indices that may be used for adaptive updates."""
    return tuple(trace_idx for trace_idx in released if int(trace_idx) >= int(start_trace_index))
