from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


OUTCOME_STRICT_CORRECT = "strict_correct"
OUTCOME_PARALLELISM_ADMISSIBLE_ERROR = "parallelism_admissible_error"
OUTCOME_OOS_ERROR = "oos_error"

REASON_UNKNOWN_PREDICTION = "unknown_prediction"
REASON_UNKNOWN_TARGET = "unknown_target"
REASON_UNRESOLVED_MAPPING = "unresolved_mask_mapping"
REASON_EMPTY_MASK = "empty_mask"


@dataclass(frozen=True)
class AuditObservation:
    prediction_identity: str | None
    target_identity: str | None
    allowed_identities: frozenset[str] | None
    prediction_space: str
    mask_space: str
    metric_contract_id: str
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class AuditClassification:
    outcome: str | None
    valid: bool
    exclusion_reason: str | None = None
    exact_outside_mask: bool = False

    @classmethod
    def unresolved(cls, reason: str) -> "AuditClassification":
        return cls(outcome=None, valid=False, exclusion_reason=reason)

    @classmethod
    def strict_correct(cls, *, exact_outside_mask: bool) -> "AuditClassification":
        return cls(
            outcome=OUTCOME_STRICT_CORRECT,
            valid=True,
            exact_outside_mask=bool(exact_outside_mask),
        )

    @classmethod
    def parallelism_admissible_error(cls) -> "AuditClassification":
        return cls(outcome=OUTCOME_PARALLELISM_ADMISSIBLE_ERROR, valid=True)

    @classmethod
    def oos_error(cls) -> "AuditClassification":
        return cls(outcome=OUTCOME_OOS_ERROR, valid=True)


@dataclass(frozen=True)
class OutcomePartitionSummary:
    audited_prefix_count: int
    valid_prediction_count: int
    strict_correct_count: int
    parallelism_admissible_error_count: int
    oos_error_count: int
    exact_outside_mask_count: int
    unknown_prediction_count: int
    unknown_target_count: int
    empty_mask_count: int
    unresolved_mapping_count: int
    excluded_count: int
    metric_contract_id: str | None
    prediction_space: str | None
    mask_space: str | None

    @property
    def strict_correct_rate(self) -> float:
        return _rate(self.strict_correct_count, self.valid_prediction_count)

    @property
    def parallelism_admissible_error_rate(self) -> float:
        return _rate(self.parallelism_admissible_error_count, self.valid_prediction_count)

    @property
    def oos_error_rate(self) -> float:
        return _rate(self.oos_error_count, self.valid_prediction_count)

    @property
    def exact_outside_mask_rate(self) -> float:
        return _rate(self.exact_outside_mask_count, self.audited_prefix_count)

    @property
    def unknown_prediction_rate(self) -> float:
        return _rate(self.unknown_prediction_count, self.audited_prefix_count)

    @property
    def unknown_target_rate(self) -> float:
        return _rate(self.unknown_target_count, self.audited_prefix_count)

    @property
    def empty_mask_rate(self) -> float:
        return _rate(self.empty_mask_count, self.audited_prefix_count)

    @property
    def unresolved_mapping_rate(self) -> float:
        return _rate(self.unresolved_mapping_count, self.audited_prefix_count)

    @property
    def excluded_rate(self) -> float:
        return _rate(self.excluded_count, self.audited_prefix_count)

    @property
    def partition_sum(self) -> float:
        if self.valid_prediction_count <= 0:
            return 0.0
        return self.strict_correct_rate + self.parallelism_admissible_error_rate + self.oos_error_rate

    def as_metrics(self) -> dict[str, float | int | str]:
        metrics: dict[str, float | int | str] = {
            "audited_prefix_count": self.audited_prefix_count,
            "valid_prediction_count": self.valid_prediction_count,
            "strict_correct_count": self.strict_correct_count,
            "parallelism_admissible_error_count": self.parallelism_admissible_error_count,
            "oos_error_count": self.oos_error_count,
            "exact_outside_mask_count": self.exact_outside_mask_count,
            "unknown_prediction_count": self.unknown_prediction_count,
            "unknown_target_count": self.unknown_target_count,
            "empty_mask_count": self.empty_mask_count,
            "unresolved_mapping_count": self.unresolved_mapping_count,
            "excluded_count": self.excluded_count,
            "strict_correct_rate": self.strict_correct_rate,
            "parallelism_admissible_error_rate": self.parallelism_admissible_error_rate,
            "oos_error_rate": self.oos_error_rate,
            "exact_outside_mask_rate": self.exact_outside_mask_rate,
            "unknown_prediction_rate": self.unknown_prediction_rate,
            "unknown_target_rate": self.unknown_target_rate,
            "empty_mask_rate": self.empty_mask_rate,
            "unresolved_mapping_rate": self.unresolved_mapping_rate,
            "excluded_rate": self.excluded_rate,
            "partition_sum": self.partition_sum,
        }
        if self.metric_contract_id is not None:
            metrics["metric_contract_id"] = self.metric_contract_id
        if self.prediction_space is not None:
            metrics["prediction_space"] = self.prediction_space
        if self.mask_space is not None:
            metrics["mask_space"] = self.mask_space
        return metrics


def classify_observation(observation: AuditObservation) -> AuditClassification:
    if observation.exclusion_reason is not None:
        return AuditClassification.unresolved(observation.exclusion_reason)
    if not observation.prediction_identity:
        return AuditClassification.unresolved(REASON_UNKNOWN_PREDICTION)
    if not observation.target_identity:
        return AuditClassification.unresolved(REASON_UNKNOWN_TARGET)
    if observation.allowed_identities is None:
        return AuditClassification.unresolved(REASON_UNRESOLVED_MAPPING)
    if not observation.allowed_identities:
        return AuditClassification.unresolved(REASON_EMPTY_MASK)
    if observation.prediction_identity == observation.target_identity:
        return AuditClassification.strict_correct(
            exact_outside_mask=observation.prediction_identity not in observation.allowed_identities
        )
    if observation.prediction_identity in observation.allowed_identities:
        return AuditClassification.parallelism_admissible_error()
    return AuditClassification.oos_error()


def aggregate_outcomes(observations: Iterable[AuditObservation]) -> OutcomePartitionSummary:
    rows = list(observations)
    contract_id = _single_value("metric_contract_id", (row.metric_contract_id for row in rows))
    prediction_space = _single_value("prediction_space", (row.prediction_space for row in rows))
    mask_space = _single_value("mask_space", (row.mask_space for row in rows))
    counts: dict[str, int] = {
        "strict_correct": 0,
        "parallelism_admissible_error": 0,
        "oos_error": 0,
        "exact_outside_mask": 0,
        "unknown_prediction": 0,
        "unknown_target": 0,
        "empty_mask": 0,
        "unresolved_mapping": 0,
        "excluded": 0,
    }

    for observation in rows:
        result = classify_observation(observation)
        if result.valid:
            if result.outcome == OUTCOME_STRICT_CORRECT:
                counts["strict_correct"] += 1
            elif result.outcome == OUTCOME_PARALLELISM_ADMISSIBLE_ERROR:
                counts["parallelism_admissible_error"] += 1
            elif result.outcome == OUTCOME_OOS_ERROR:
                counts["oos_error"] += 1
            else:
                raise ValueError(f"Unknown RS-01 outcome: {result.outcome!r}")
            if result.exact_outside_mask:
                counts["exact_outside_mask"] += 1
            continue

        counts["excluded"] += 1
        reason = result.exclusion_reason
        if reason == REASON_UNKNOWN_PREDICTION:
            counts["unknown_prediction"] += 1
        elif reason == REASON_UNKNOWN_TARGET:
            counts["unknown_target"] += 1
        elif reason == REASON_EMPTY_MASK:
            counts["empty_mask"] += 1
        else:
            counts["unresolved_mapping"] += 1

    valid_count = counts["strict_correct"] + counts["parallelism_admissible_error"] + counts["oos_error"]
    summary = OutcomePartitionSummary(
        audited_prefix_count=len(rows),
        valid_prediction_count=valid_count,
        strict_correct_count=counts["strict_correct"],
        parallelism_admissible_error_count=counts["parallelism_admissible_error"],
        oos_error_count=counts["oos_error"],
        exact_outside_mask_count=counts["exact_outside_mask"],
        unknown_prediction_count=counts["unknown_prediction"],
        unknown_target_count=counts["unknown_target"],
        empty_mask_count=counts["empty_mask"],
        unresolved_mapping_count=counts["unresolved_mapping"],
        excluded_count=counts["excluded"],
        metric_contract_id=contract_id,
        prediction_space=prediction_space,
        mask_space=mask_space,
    )
    if valid_count != (
        summary.strict_correct_count
        + summary.parallelism_admissible_error_count
        + summary.oos_error_count
    ):
        raise ValueError("RS-01 valid partition does not sum to valid_prediction_count.")
    if valid_count > 0 and abs(summary.partition_sum - 1.0) > 1.0e-12:
        raise ValueError("RS-01 partition rates do not sum to 1.0.")
    return summary


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _single_value(field: str, values: Iterable[str]) -> str | None:
    unique = {value for value in values if value}
    if len(unique) > 1:
        raise ValueError(f"Mixed {field} values are not allowed in one RS-01 partition.")
    return next(iter(unique), None)
