from __future__ import annotations

from dataclasses import replace

import pytest

from src.application.services.outcome_partition_audit import (
    AuditObservation,
    aggregate_outcomes,
    classify_observation,
)


def _observation(*, prediction: str, target: str, allowed: set[str]) -> AuditObservation:
    return AuditObservation(
        prediction_identity=prediction,
        target_identity=target,
        allowed_identities=frozenset(allowed),
        prediction_space="label",
        mask_space="label",
        metric_contract_id="rs01.label_mask.v1",
    )


def test_strict_correct_precedes_mask_membership_and_records_exact_outside_mask():
    result = classify_observation(_observation(prediction="A", target="A", allowed={"B"}))

    assert result.outcome == "strict_correct"
    assert result.exact_outside_mask is True


def test_partition_is_mutually_exclusive_and_exhaustive_for_valid_rows():
    summary = aggregate_outcomes(
        [
            _observation(prediction="A", target="A", allowed={"A"}),
            _observation(prediction="B", target="A", allowed={"B"}),
            _observation(prediction="C", target="A", allowed={"B"}),
        ]
    )

    assert summary.valid_prediction_count == 3
    assert summary.strict_correct_count == 1
    assert summary.parallelism_admissible_error_count == 1
    assert summary.oos_error_count == 1
    assert summary.partition_sum == 1.0


def test_unknown_and_empty_mask_rows_are_excluded_from_valid_partition():
    summary = aggregate_outcomes(
        [
            _observation(prediction="A", target="A", allowed={"A"}),
            _observation(prediction="", target="A", allowed={"A"}),
            _observation(prediction="A", target="", allowed={"A"}),
            _observation(prediction="A", target="A", allowed=set()),
        ]
    )

    assert summary.audited_prefix_count == 4
    assert summary.valid_prediction_count == 1
    assert summary.unknown_prediction_count == 1
    assert summary.unknown_target_count == 1
    assert summary.empty_mask_count == 1
    assert summary.excluded_count == 3
    assert summary.partition_sum == 1.0


def test_mixed_metric_contracts_fail_closed():
    first = _observation(prediction="A", target="A", allowed={"A"})
    second = _observation(prediction="A", target="A", allowed={"A"})
    second = replace(second, metric_contract_id="rs01.other.v1")

    with pytest.raises(ValueError, match="metric_contract_id"):
        aggregate_outcomes([first, second])
