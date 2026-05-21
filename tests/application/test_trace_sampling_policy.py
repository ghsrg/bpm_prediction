from __future__ import annotations

import json

from src.application.ports.trace_recorder_port import StructuralTraceEvent
from src.application.services.trace_sampling_policy import TraceSamplingPolicy, classify_trace_reason


def test_trace_event_payload_is_plain_json_serializable():
    event = StructuralTraceEvent(
        name="structural_prediction_debug",
        inputs={"sample": {"trace_idx": 1}},
        outputs={"prediction": {"pred_index": 2}},
        attributes={"stage": "eval_drift_one_pass", "strict_correct": False},
    )

    encoded = json.dumps(event.to_dict())

    assert "structural_prediction_debug" in encoded


def test_trace_sampling_policy_respects_stage_version_and_run_limits():
    policy = TraceSamplingPolicy(
        enabled=True,
        stages={"eval_drift_one_pass"},
        max_traces_per_run=3,
        max_traces_per_stage=2,
        max_traces_per_version=1,
    )

    assert policy.should_record(stage="eval_drift_one_pass", version="v1", reason="version_first")
    assert not policy.should_record(
        stage="eval_drift_one_pass",
        version="v1",
        reason="strict_error_but_allowed",
    )
    assert policy.should_record(stage="eval_drift_one_pass", version="v2", reason="version_first")
    assert not policy.should_record(stage="inference", version="v3", reason="version_first")


def test_trace_sampling_policy_selects_interesting_errors_before_random():
    reason = classify_trace_reason(
        strict_correct=False,
        pred_in_mask=True,
        target_in_mask=True,
        confidence=0.9,
        version_seen=False,
        high_confidence_threshold=0.8,
        low_confidence_threshold=0.4,
    )

    assert reason == "strict_error_but_allowed"


def test_trace_sampling_policy_can_select_first_sample_per_version():
    reason = classify_trace_reason(
        strict_correct=True,
        pred_in_mask=True,
        target_in_mask=True,
        confidence=0.9,
        version_seen=False,
        high_confidence_threshold=0.8,
        low_confidence_threshold=0.4,
    )

    assert reason == "version_first"
