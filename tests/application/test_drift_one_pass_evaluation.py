from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.application.use_cases.trainer import ModelTrainer
from src.domain.entities.event_record import EventRecord
from src.domain.entities.raw_trace import RawTrace


class _FailOnReadAdapter:
    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        _ = file_path
        _ = mapping_config
        raise AssertionError("Adapter read should not be called in one-pass drift tests.")


class _NoopPrefixPolicy:
    def generate_slices(self, trace: RawTrace):
        _ = trace
        return []


class _NoopGraphBuilder:
    def build_graph(self, prefix_slice):
        _ = prefix_slice
        raise AssertionError("Graph builder should not be called in one-pass drift tests.")


class _PredictFromXNumModel(nn.Module):
    def __init__(self, output_dim: int = 3) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, contract):
        pred = contract["x_num"].view(-1).long().clamp(min=0, max=self.output_dim - 1)
        logits = torch.full((int(pred.shape[0]), self.output_dim), -5.0, device=pred.device)
        logits[torch.arange(int(pred.shape[0]), device=pred.device), pred] = 5.0
        return logits


class _FakeTracker:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, int | None]] = []

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.metrics.append((key, float(value), step))

    def log_param(self, key: str, value) -> None:
        _ = key
        _ = value

    def log_tag(self, key: str, value) -> None:
        _ = key
        _ = value

    def log_artifact(self, local_path: str) -> None:
        _ = local_path

    def log_model(self, model, artifact_path: str) -> None:
        _ = model
        _ = artifact_path


def _event(idx: int) -> EventRecord:
    return EventRecord(
        activity_id=f"A{idx}",
        timestamp=float(1700000000 + idx),
        resource_id="r1",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": f"A{idx}", "org:resource": "r1"},
        activity_instance_id=f"ai_{idx}",
    )


def _trace(case_id: str, idx: int) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version="v1",
        events=[_event(idx), _event(idx + 1)],
        trace_attributes={},
    )


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
        "process_version_idx": torch.tensor([0], dtype=torch.long),
        "trace_start_ts": torch.tensor([float(1700000000 + trace_idx)], dtype=torch.float64),
        "trace_end_ts": torch.tensor([float(1700000010 + trace_idx)], dtype=torch.float64),
    }
    if mask is not None:
        payload["allowed_target_mask"] = torch.tensor([mask], dtype=torch.bool)
    return Data(**payload)


def _trainer(tmp_path, *, drift_window_size: int = 2, drift_window_sliding: int = 1) -> ModelTrainer:
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
            "drift_window_size": drift_window_size,
            "drift_window_sliding": drift_window_sliding,
            "experiment_config": {
                "name": "pytest_one_pass_drift",
                "mode": "eval_drift",
                "drift_window_size": drift_window_size,
                "drift_window_sliding": drift_window_sliding,
            },
        },
        prepared_data={"idx_to_version": {0: "v1"}},
    )


def test_collect_drift_inference_records_is_compact(tmp_path):
    trainer = _trainer(tmp_path)
    loader = DataLoader(
        [
            _sample(trace_idx=0, target=1, pred=1, mask=[False, True, False]),
            _sample(trace_idx=1, target=2, pred=1, mask=[False, True, True]),
        ],
        batch_size=2,
        shuffle=False,
    )

    records = trainer._collect_drift_inference_records(loader)

    assert records.trace_idx.tolist() == [0, 1]
    assert records.y_true.tolist() == [1, 2]
    assert records.y_pred.tolist() == [1, 1]
    assert records.confidence.shape == (2,)
    assert records.target_in_mask_flags.tolist() == pytest.approx([1.0, 1.0])
    assert records.pred_in_mask_flags.tolist() == pytest.approx([1.0, 1.0])
    assert not hasattr(records, "y_prob")


def test_record_metrics_match_evaluate_test_for_full_dataset(tmp_path):
    trainer = _trainer(tmp_path)
    samples = [
        _sample(trace_idx=0, target=1, pred=1, mask=[False, True, False]),
        _sample(trace_idx=1, target=2, pred=1, mask=[False, True, True]),
        _sample(trace_idx=2, target=0, pred=2, mask=[True, False, True]),
    ]

    legacy_metrics = trainer._evaluate_test(DataLoader(samples, batch_size=3, shuffle=False), stage_label="eval_drift")
    records = trainer._collect_drift_inference_records(DataLoader(samples, batch_size=3, shuffle=False))
    new_metrics = trainer._compute_test_metrics_from_records(records, np.arange(records.y_true.shape[0]))

    for key in [
        "test_macro_f1",
        "strict_test_macro_f1",
        "test_accuracy",
        "strict_test_accuracy",
        "test_top3_accuracy",
        "test_oos",
        "test_target_in_mask_rate",
        "test_pred_in_mask_rate",
        "test_strict_error_but_allowed_rate",
        "test_ambiguous_prefix_rate",
    ]:
        assert new_metrics[key] == pytest.approx(legacy_metrics[key])


def test_resolve_drift_window_record_indices_uses_trace_idx_ranges(tmp_path):
    trainer = _trainer(tmp_path, drift_window_size=2, drift_window_sliding=1)
    samples = [
        _sample(trace_idx=0, target=1, pred=1),
        _sample(trace_idx=0, target=1, pred=1),
        _sample(trace_idx=1, target=1, pred=1),
        _sample(trace_idx=2, target=1, pred=1),
        _sample(trace_idx=2, target=1, pred=1),
        _sample(trace_idx=3, target=1, pred=1),
    ]
    records = trainer._collect_drift_inference_records(DataLoader(samples, batch_size=6, shuffle=False))
    traces = [_trace(f"c{idx}", idx) for idx in range(4)]

    windows = trainer._resolve_drift_window_record_indices(records, traces)

    assert [item[1].tolist() for item in windows] == [[0, 1, 2], [2, 3, 4], [3, 4, 5]]


def test_run_eval_drift_uses_one_pass_prebuilt_dataset_without_legacy_graph_rebuild(tmp_path):
    trainer = _trainer(tmp_path, drift_window_size=2, drift_window_sliding=1)
    samples = [
        _sample(trace_idx=0, target=1, pred=1, mask=[False, True, False]),
        _sample(trace_idx=1, target=2, pred=1, mask=[False, True, True]),
        _sample(trace_idx=2, target=0, pred=2, mask=[True, False, True]),
        _sample(trace_idx=3, target=1, pred=1, mask=[False, True, False]),
    ]
    traces = [_trace(f"c{idx}", idx) for idx in range(4)]

    def _fail_legacy_build_loader(window_traces, shuffle):
        _ = window_traces
        _ = shuffle
        raise AssertionError("legacy graph rebuild path should not be called")

    trainer._build_loader = _fail_legacy_build_loader  # type: ignore[method-assign]

    result = trainer._run_eval_drift(
        split_data=type("Split", (), {"train": [], "val": [], "test": traces})(),
        drift_traces=traces,
        best_epoch=1,
        best_val_loss=0.5,
        prebuilt_test_dataset=samples,
    )

    assert result["mode"] == "eval_drift"
    assert len(result["drift_metrics"]) == 3


def test_run_eval_drift_falls_back_when_prebuilt_dataset_lacks_trace_idx(tmp_path):
    trainer = _trainer(tmp_path, drift_window_size=2, drift_window_sliding=1)
    samples = [Data(y=torch.tensor([1], dtype=torch.long))]
    traces = [_trace(f"c{idx}", idx) for idx in range(2)]

    trainer._evaluate_drift_windows = lambda drift_traces: [{"window_index": 0.0}]  # type: ignore[method-assign]

    result = trainer._run_eval_drift(
        split_data=type("Split", (), {"train": [], "val": [], "test": traces})(),
        drift_traces=traces,
        best_epoch=1,
        best_val_loss=0.5,
        prebuilt_test_dataset=samples,
    )

    assert result["drift_metrics"] == [{"window_index": 0.0}]


def test_one_pass_drift_rows_preserve_legacy_output_keys(tmp_path):
    trainer = _trainer(tmp_path, drift_window_size=2, drift_window_sliding=1)
    samples = [
        _sample(trace_idx=0, target=1, pred=1, mask=[False, True, False]),
        _sample(trace_idx=1, target=2, pred=1, mask=[False, True, True]),
    ]
    traces = [_trace(f"c{idx}", idx) for idx in range(2)]

    rows = trainer._evaluate_drift_windows_from_prebuilt_dataset(traces, samples)

    assert rows is not None
    assert rows
    expected_keys = {
        "window_index",
        "window_start_trace",
        "window_end_trace",
        "window_start_ts",
        "window_end_ts",
        "window_macro_f1",
        "window_strict_macro_f1",
        "window_test_ece",
        "window_test_set_nll",
        "window_test_oos",
        "window_target_in_mask_rate",
        "window_pred_in_mask_rate",
        "window_strict_error_but_allowed_rate",
        "window_ambiguous_prefix_rate",
    }
    assert expected_keys.issubset(rows[0].keys())


def test_one_pass_drift_logs_legacy_tracker_metric_names(tmp_path):
    tracker = _FakeTracker()
    trainer = _trainer(tmp_path, drift_window_size=2, drift_window_sliding=1)
    trainer.tracker = tracker
    samples = [
        _sample(trace_idx=0, target=1, pred=1, mask=[False, True, False]),
        _sample(trace_idx=1, target=2, pred=1, mask=[False, True, True]),
    ]
    traces = [_trace(f"c{idx}", idx) for idx in range(2)]

    rows = trainer._evaluate_drift_windows_from_prebuilt_dataset(traces, samples)

    assert rows is not None
    logged_names = {name for name, _, _ in tracker.metrics}
    assert {
        "drift_window_macro_f1",
        "drift_window_strict_macro_f1",
        "drift_window_test_ece",
        "drift_window_test_set_nll",
        "drift_window_test_oos",
        "drift_window_target_in_mask_rate",
        "drift_window_pred_in_mask_rate",
        "drift_window_strict_error_but_allowed_rate",
        "drift_window_ambiguous_prefix_rate",
        "drift_window_start_ts",
        "drift_window_end_ts",
    }.issubset(logged_names)
