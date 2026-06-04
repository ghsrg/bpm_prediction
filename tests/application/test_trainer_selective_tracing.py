from __future__ import annotations

from typing import Iterator

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.application.ports.trace_recorder_port import StructuralTraceEvent
from src.application.use_cases.trainer import ModelTrainer
from src.domain.entities.raw_trace import RawTrace


class _FailOnReadAdapter:
    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        _ = file_path
        _ = mapping_config
        raise AssertionError("Adapter read should not be called.")


class _NoopPrefixPolicy:
    def generate_slices(self, trace: RawTrace):
        _ = trace
        return []


class _NoopGraphBuilder:
    def build_graph(self, prefix_slice):
        _ = prefix_slice
        raise AssertionError("Graph builder should not be called.")


class _PredictFromXNumModel(nn.Module):
    structural_mode = True
    fusion_mode = "StructXAttn"

    def __init__(self, output_dim: int = 3) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.dummy = nn.Parameter(torch.tensor(0.0))
        self.last_struct_xattn_to_observed_ratio = torch.tensor(0.25)
        self.last_struct_xattn_attention_entropy = torch.tensor(1.5)
        self.last_struct_xattn_gate_mean = torch.tensor(0.2)

    def forward(self, contract):
        pred = contract["x_num"].view(-1).long().clamp(min=0, max=self.output_dim - 1)
        logits = torch.full((int(pred.shape[0]), self.output_dim), -5.0, device=pred.device)
        logits[torch.arange(int(pred.shape[0]), device=pred.device), pred] = 5.0
        return logits + (self.dummy * 0.0)


class _RecordingTraceRecorder:
    def __init__(self) -> None:
        self.events: list[StructuralTraceEvent] = []

    def record(self, event: StructuralTraceEvent) -> None:
        self.events.append(event)


def _sample(*, trace_idx: int, target: int, pred: int, mask: list[bool] | None = None) -> Data:
    payload = {
        "x_cat": torch.zeros((1, 1), dtype=torch.long),
        "x_num": torch.tensor([[float(pred)]], dtype=torch.float32),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "edge_type": torch.empty((0,), dtype=torch.long),
        "y": torch.tensor([target], dtype=torch.long),
        "num_nodes": 1,
        "trace_idx": torch.tensor([trace_idx], dtype=torch.long),
        "prefix_idx": torch.tensor([0], dtype=torch.long),
        "prefix_len": torch.tensor([1], dtype=torch.long),
        "process_version_idx": torch.tensor([1], dtype=torch.long),
    }
    if mask is not None:
        payload["allowed_target_mask"] = torch.tensor([mask], dtype=torch.bool)
    return Data(**payload)


def _trainer(tmp_path, *, trace_recorder=None, tracing_enabled: bool = True) -> ModelTrainer:
    return ModelTrainer(
        xes_adapter=_FailOnReadAdapter(),
        prefix_policy=_NoopPrefixPolicy(),  # type: ignore[arg-type]
        graph_builder=_NoopGraphBuilder(),  # type: ignore[arg-type]
        model=_PredictFromXNumModel(output_dim=3),  # type: ignore[arg-type]
        log_path="in_memory.xes",
        config={
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "checkpoint_dir": str(tmp_path),
            "tracking_config": {
                "tracing": {
                    "enabled": tracing_enabled,
                    "stages": ["inference", "eval_drift_one_pass"],
                    "sample_policy": "interesting",
                    "max_traces_per_run": 10,
                    "max_traces_per_stage": 10,
                    "max_traces_per_version": 10,
                    "top_k": 2,
                }
            },
        },
        prepared_data={
            "idx_to_version": {1: "v2"},
            "reverse_activity_vocab": {0: "<UNK>", 1: "Approve", 2: "Reject"},
        },
        trace_recorder=trace_recorder,
    )


def assert_no_tensors(value):
    if isinstance(value, torch.Tensor):
        raise AssertionError("trace event must not contain torch.Tensor")
    if isinstance(value, dict):
        for item in value.values():
            assert_no_tensors(item)
    if isinstance(value, list):
        for item in value:
            assert_no_tensors(item)


def test_evaluate_test_records_selected_structural_trace(tmp_path):
    recorder = _RecordingTraceRecorder()
    trainer = _trainer(tmp_path, trace_recorder=recorder)
    loader = DataLoader(
        [_sample(trace_idx=7, target=2, pred=1, mask=[False, True, True])],
        batch_size=1,
        shuffle=False,
    )

    trainer._evaluate_test(loader, stage_label="inference")

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event.attributes["stage"] == "inference"
    assert event.attributes["reason"] == "strict_error_but_allowed"
    assert event.attributes["fusion_mode"] == "StructXAttn"
    assert event.attributes["process_version"] == "v2"
    assert event.outputs["prediction"]["target_label"] == "Reject"
    assert_no_tensors(event.to_dict())


def test_disabled_tracing_records_nothing(tmp_path):
    recorder = _RecordingTraceRecorder()
    trainer = _trainer(tmp_path, trace_recorder=recorder, tracing_enabled=False)
    loader = DataLoader(
        [_sample(trace_idx=7, target=2, pred=1, mask=[False, True, True])],
        batch_size=1,
        shuffle=False,
    )

    trainer._evaluate_test(loader, stage_label="inference")

    assert recorder.events == []


def test_collect_drift_inference_records_records_selected_trace_without_changing_records(tmp_path):
    recorder = _RecordingTraceRecorder()
    trainer = _trainer(tmp_path, trace_recorder=recorder)
    loader = DataLoader(
        [_sample(trace_idx=7, target=2, pred=1, mask=[False, True, True])],
        batch_size=1,
        shuffle=False,
    )

    records = trainer._collect_drift_inference_records(loader)

    assert records.trace_idx.tolist() == [7]
    assert records.y_true.tolist() == ["Reject"]
    assert records.y_pred.tolist() == ["Approve"]
    assert records.fixed_y_true.tolist() == [2]
    assert records.fixed_y_pred.tolist() == [1]
    assert len(recorder.events) == 1
    assert recorder.events[0].attributes["stage"] == "eval_drift_one_pass"
