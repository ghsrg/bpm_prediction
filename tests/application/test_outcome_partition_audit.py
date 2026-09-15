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


def _common(prediction, target, allowed):
    return replace(
        _observation(prediction=prediction, target=target, allowed=allowed),
        metric_contract_id="state_aware_activity_label_mask.v2",
    )


def test_common_empty_masks_are_valid_and_unseen_targets_are_observed():
    summary = aggregate_outcomes([
        _common("A", "A", set()),
        _common("B", "unseen_observed_activity", set()),
        _common("B", "unseen_observed_activity", {"B"}),
    ])
    assert summary.valid_prediction_count == 3
    assert summary.excluded_count == 0
    assert summary.empty_mask_count == 2
    assert summary.strict_correct_count == 1
    assert summary.oos_error_count == 1
    assert summary.parallelism_admissible_error_count == 1


def test_common_safety_rates_share_denominator_with_partial_coverage():
    summary = aggregate_outcomes([
        _common("A", "A", {"B"}),
        _common("B", "A", {"A"}),
        _common("B", "A", {"B"}),
        replace(_common("B", "A", set()), allowed_identities=None),
    ])
    metrics = summary.as_metrics()
    assert metrics["common_safety_denominator_count"] == 3
    assert metrics["exact_outside_mask_rate"] == pytest.approx(1 / 3)
    assert metrics["common_oos_rate"] == pytest.approx(2 / 3)
    assert metrics["common_oos_rate"] == pytest.approx(
        metrics["oos_error_rate"] + metrics["exact_outside_mask_rate"]
    )
    assert metrics["common_target_in_mask_count"] == 1
    assert metrics["common_target_in_mask_rate"] == pytest.approx(1 / 3)
    assert metrics["audit_coverage"] == pytest.approx(3 / 4)


def test_common_abstention_is_distinct_from_unknown_prediction_mapping():
    summary = aggregate_outcomes([
        replace(_common(None, "A", {"A"}), exclusion_reason="abstention"),
        _common(None, "A", {"A"}),
        _common("B", "A", {"A"}),
    ])
    metrics = summary.as_metrics()
    assert metrics["abstention_count"] == 1
    assert metrics["unknown_prediction_count"] == 1
    assert metrics["excluded_count"] == 2


@pytest.mark.parametrize(
    ("prediction", "target", "allowed", "outcome", "exact_outside"),
    [
        ("A", "A", {"A", "B"}, "strict_correct", False),
        ("B", "A", {"B"}, "parallelism_admissible_error", False),
        ("C", "A", {"B"}, "oos_error", False),
        ("A", "A", {"B"}, "strict_correct", True),
        ("B", "A", set(), "oos_error", False),
        ("A", "A", set(), "strict_correct", True),
    ],
)
def test_common_outcome_truth_table(prediction, target, allowed, outcome, exact_outside):
    result = classify_observation(_common(prediction, target, allowed))
    assert result.valid is True
    assert result.outcome == outcome
    assert result.exact_outside_mask is exact_outside


def test_common_partition_count_rate_and_pred_in_mask_invariants():
    summary = aggregate_outcomes([
        _common("A", "A", {"A", "B"}),
        _common("B", "A", {"B"}),
        _common("C", "A", {"B"}),
        _common("A", "A", {"B"}),
        _common("B", "A", set()),
        _common("A", "A", set()),
    ])
    metrics = summary.as_metrics()
    assert summary.valid_prediction_count == summary.audited_prefix_count == 6
    assert summary.excluded_count == 0
    assert summary.strict_correct_count + summary.parallelism_admissible_error_count + summary.oos_error_count == 6
    assert summary.strict_correct_rate == pytest.approx(summary.strict_correct_count / 6)
    assert metrics["common_oos_rate"] == pytest.approx(
        metrics["oos_error_rate"] + metrics["exact_outside_mask_rate"]
    )
    assert metrics["common_pred_in_mask_rate"] == pytest.approx(1 - metrics["common_oos_rate"])
