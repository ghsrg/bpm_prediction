from __future__ import annotations

import torch

from src.application.ports.trace_recorder_port import StructuralTraceEvent
from src.application.services.structural_trace_payload_builder import (
    build_structural_prediction_trace_event,
    build_structural_prediction_trace_payload,
)


class _ObservedOnlyModel:
    structural_mode = False
    fusion_mode = "ClassMeanConcat"


class _ClassAwareModel:
    structural_mode = True
    fusion_mode = "ClassAwareStructuralScoring"

    def __init__(self) -> None:
        self.last_observed_logits = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)
        self.last_structural_class_logits = torch.tensor([[0.1, -0.2, 0.3]], requires_grad=True)


class _StructXAttnModel:
    structural_mode = True
    fusion_mode = "StructXAttn"

    def __init__(self) -> None:
        self.last_struct_xattn_to_observed_ratio = torch.tensor(0.25, requires_grad=True)
        self.last_struct_xattn_attention_entropy = torch.tensor(1.5, requires_grad=True)
        self.last_struct_xattn_gate_mean = torch.tensor(0.2, requires_grad=True)


def _contract() -> dict:
    return {
        "x_cat": torch.zeros((2, 1), dtype=torch.long),
        "x_num": torch.ones((2, 1), dtype=torch.float32),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "batch": torch.zeros((2,), dtype=torch.long),
        "struct_x": torch.ones((3, 2), dtype=torch.float32),
        "structural_edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "allowed_target_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        "stats_snapshot_versions": ["k000001"],
        "stats_snapshot_as_of_ts_batch": ["2026-03-10T00:00:00+00:00"],
        "stats_allowed_batch": [True],
        "stats_missing_asof_snapshot_batch": [False],
    }


def assert_no_tensors(value):
    if isinstance(value, torch.Tensor):
        raise AssertionError("trace payload must not contain torch.Tensor")
    if isinstance(value, dict):
        for item in value.values():
            assert_no_tensors(item)
    if isinstance(value, list):
        for item in value:
            assert_no_tensors(item)


def test_payload_builder_handles_observed_only_model():
    logits = torch.tensor([[0.1, 2.0, 0.3]], requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    payload = build_structural_prediction_trace_payload(
        stage="inference",
        global_index=0,
        contract=_contract(),
        logits=logits,
        effective_logits=logits,
        probs=probs,
        targets=torch.tensor([1]),
        predictions=torch.tensor([1]),
        model=_ObservedOnlyModel(),
        reverse_activity_vocab={0: "<UNK>", 1: "Approve", 2: "Reject"},
        row_index=0,
        reason="version_first",
        top_k=2,
    )

    assert payload["run"]["structural_mode"] is False
    assert payload["prediction"]["target_label"] == "Approve"
    assert payload["prediction"]["pred_label"] == "Approve"
    assert len(payload["prediction"]["top_k"]) == 2
    assert payload["diagnostics"]["generic"]["prediction_entropy"] >= 0.0
    assert_no_tensors(payload)


def test_payload_builder_includes_class_aware_structural_logits():
    logits = torch.tensor([[2.1, -1.2, 0.8]], requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    payload = build_structural_prediction_trace_payload(
        stage="eval_drift_one_pass",
        global_index=3,
        contract=_contract(),
        logits=logits,
        effective_logits=logits,
        probs=probs,
        targets=torch.tensor([2]),
        predictions=torch.tensor([0]),
        model=_ClassAwareModel(),
        reverse_activity_vocab={0: "<UNK>", 1: "Approve", 2: "Reject"},
        row_index=0,
        reason="strict_error_but_allowed",
        top_k=3,
    )

    diagnostics = payload["diagnostics"]["class_aware_structural_scoring"]
    assert diagnostics["target_structural_logit"] == 0.3
    assert diagnostics["pred_observed_logit"] == 2.0
    assert diagnostics["structural_to_observed_logit_ratio"] > 0.0
    assert_no_tensors(payload)


def test_payload_builder_includes_struct_xattn_diagnostics():
    logits = torch.tensor([[0.1, 0.2, 2.0]], requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    payload = build_structural_prediction_trace_payload(
        stage="eval_drift_one_pass",
        global_index=4,
        contract=_contract(),
        logits=logits,
        effective_logits=logits,
        probs=probs,
        targets=torch.tensor([1]),
        predictions=torch.tensor([2]),
        model=_StructXAttnModel(),
        reverse_activity_vocab={0: "<UNK>", 1: "Approve", 2: "Reject"},
        row_index=0,
        reason="strict_error_but_allowed",
        top_k=2,
    )

    diagnostics = payload["diagnostics"]["struct_xattn"]
    assert diagnostics["struct_xattn_to_observed_ratio"] == 0.25
    assert diagnostics["struct_xattn_attention_entropy"] == 1.5
    assert_no_tensors(payload)


def test_trace_event_uses_flat_searchable_attributes():
    logits = torch.tensor([[0.1, 0.2, 2.0]], requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    event = build_structural_prediction_trace_event(
        stage="eval_drift_one_pass",
        global_index=4,
        contract=_contract(),
        logits=logits,
        effective_logits=logits,
        probs=probs,
        targets=torch.tensor([1]),
        predictions=torch.tensor([2]),
        model=_StructXAttnModel(),
        reverse_activity_vocab={0: "<UNK>", 1: "Approve", 2: "Reject"},
        row_index=0,
        reason="strict_error_but_allowed",
        top_k=2,
    )

    assert isinstance(event, StructuralTraceEvent)
    assert event.attributes["fusion_mode"] == "StructXAttn"
    assert event.attributes["reason"] == "strict_error_but_allowed"
    assert event.attributes["strict_correct"] is False
    assert event.attributes["target_in_mask"] is True
    assert event.attributes["pred_in_mask"] is True
    assert event.attributes["prediction_in_mask"] is True
    assert event.attributes["process_version"] == "__unknown__"
    assert all("." not in key for key in event.attributes)
    assert_no_tensors(event.to_dict())
