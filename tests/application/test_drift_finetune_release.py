from __future__ import annotations

import pytest

from src.application.services.drift_release_policy import (
    adaptive_window_start_trace,
    eligible_released_trace_indices,
    newly_observed_trace_indices,
    released_trace_indices,
)


def test_adaptive_window_start_aligns_cut_to_next_regular_boundary():
    assert adaptive_window_start_trace(cut_trace=38, window_step=10) == 40


def test_alignment_partitions_but_does_not_remove_evaluation_windows():
    starts = (0, 10, 20, 30, 40, 50)
    adaptive_start = adaptive_window_start_trace(cut_trace=38, window_step=10)

    assert tuple(start for start in starts if start < adaptive_start) == (0, 10, 20, 30)
    assert tuple(start for start in starts if start >= adaptive_start) == (40, 50)


def test_observed_stride_is_the_prefix_that_leaves_next_forecast_window():
    current = tuple(range(40, 140))
    following = tuple(range(50, 150))

    assert newly_observed_trace_indices(current, following, frozenset()) == tuple(range(40, 50))


def test_observed_stride_never_reuses_an_already_trained_trace():
    current = tuple(range(40, 140))
    following = tuple(range(50, 150))

    assert newly_observed_trace_indices(current, following, frozenset(range(40, 45))) == tuple(range(45, 50))


def test_release_excludes_next_overlapping_window():
    current = tuple(range(100))
    following = tuple(range(10, 110))

    released = released_trace_indices(current, following, frozenset())

    assert released == tuple(range(10))
    assert set(released).isdisjoint(following)


def test_release_skips_seen_indices_and_preserves_order():
    current = (10, 11, 12, 13, 14)
    following = (13, 14, 15, 16, 17)
    seen = frozenset({10, 12})

    assert released_trace_indices(current, following, seen) == (11,)


def test_tumbling_window_releases_full_previous_window():
    current = (1, 2, 3)
    following = (4, 5, 6)

    assert released_trace_indices(current, following, frozenset()) == current


def test_gap_between_windows_is_not_released():
    current = (1, 2, 3)
    following = (8, 9, 10)

    assert released_trace_indices(current, following, frozenset()) == current


def test_empty_release_when_next_window_fully_overlaps_current():
    current = (1, 2, 3)
    following = (1, 2, 3)

    assert released_trace_indices(current, following, frozenset()) == ()


def test_finetune_start_ratio_filters_released_indices_before_cutoff():
    released = (0, 1, 2, 3, 4)

    eligible = eligible_released_trace_indices(released, start_trace_index=2)

    assert eligible == (2, 3, 4)


@pytest.mark.parametrize(
    ("current", "following"),
    [
        ((1, 1, 2), (3, 4, 5)),
        ((1, 2, 3), (4, 4, 5)),
    ],
)
def test_rejects_duplicate_stream_indices(current, following):
    with pytest.raises(ValueError, match="duplicate stream indices"):
        released_trace_indices(current, following, frozenset())
