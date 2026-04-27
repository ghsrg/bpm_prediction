from __future__ import annotations

from typing import Iterator

import torch
from torch_geometric.data import Data

from src.application.use_cases.trainer import ModelTrainer
from src.cli import (
    _apply_cascade_prepare,
    _format_trace_version_counts,
    _tensor_scale_diagnostics,
    _trace_version_counts,
)
from src.domain.entities.event_record import EventRecord
from src.domain.entities.feature_config import FeatureLayout
from src.domain.entities.raw_trace import RawTrace
from src.domain.models.baseline_gcn import BaselineGCN


class _DummyAdapter:
    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        _ = file_path
        _ = mapping_config
        return iter([])


class _DummyPrefixPolicy:
    def generate_slices(self, trace: RawTrace):
        _ = trace
        return []


class _DummyGraphBuilder:
    def build_graph(self, prefix):
        _ = prefix
        raise RuntimeError("Not used in cascade split unit tests.")


def _build_trace(ts: float, case_id: str, version: str = "v1") -> RawTrace:
    event = EventRecord(
        activity_id="A",
        timestamp=ts,
        resource_id="R",
        lifecycle="complete",
        position_in_trace=0,
        duration=1.0,
        time_since_case_start=0.0,
        time_since_previous_event=0.0,
        extra={"concept:name": "A"},
        activity_instance_id=f"ai_{case_id}",
    )
    return RawTrace(case_id=case_id, process_version=version, events=[event], trace_attributes={})


def _trainer(experiment_config: dict, mode: str = "train") -> ModelTrainer:
    model = BaselineGCN(
        feature_layout=FeatureLayout(cat_features={}, cat_feature_names=[], num_dim=1),
        hidden_dim=8,
        output_dim=3,
        dropout=0.0,
    )
    return ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": mode,
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "experiment_config": experiment_config,
        },
    )


def test_cascade_split_temporal_train_ratio_fraction_then_micro_split():
    traces = [_build_trace(ts=float(ts), case_id=f"c{idx}") for idx, ts in enumerate([5, 1, 4, 0, 3, 2, 9, 7, 8, 6])]
    trainer = _trainer(
        experiment_config={
            "split_strategy": "temporal",
            "train_ratio": 0.7,
            "fraction": 0.5,
            "split_ratio": [0.5, 0.25, 0.25],
        },
        mode="train",
    )

    prepared = trainer._prepare_data(traces, mode="train")
    prepared_ts = [trace.events[0].timestamp for trace in prepared]
    assert prepared_ts == [0.0, 1.0, 2.0]

    split = trainer._prepare_split_data(prepared)
    assert len(split.train) == 1
    assert len(split.val) == 0
    assert len(split.test) == 2


def test_cascade_split_eval_drift_uses_tail_only_after_macro_cut():
    traces = [_build_trace(ts=float(ts), case_id=f"c{idx}") for idx, ts in enumerate(range(10))]
    trainer = _trainer(
        experiment_config={
            "split_strategy": "temporal",
            "train_ratio": 0.7,
            "fraction": 0.5,
            "split_ratio": [0.5, 0.25, 0.25],
        },
        mode="eval_drift",
    )

    prepared = trainer._prepare_data(traces, mode="eval_drift")
    prepared_ts = [trace.events[0].timestamp for trace in prepared]
    assert prepared_ts == [7.0]


def test_cascade_split_none_preserves_order_and_handles_small_fraction_without_index_errors():
    traces = [_build_trace(ts=float(ts), case_id=f"c{idx}") for idx, ts in enumerate([5, 1, 4, 0, 3, 2])]
    trainer = _trainer(
        experiment_config={
            "split_strategy": "none",
            "train_ratio": 0.5,
            "fraction": 0.01,
            "split_ratio": [0.7, 0.2, 0.1],
        },
        mode="train",
    )

    prepared = trainer._prepare_data(traces, mode="train")
    assert prepared == []

    split = trainer._prepare_split_data(prepared)
    assert len(split.train) == 0
    assert len(split.val) == 0
    assert len(split.test) == 0


def test_cascade_prepare_diagnostics_detect_single_version_cut_from_versioned_log():
    traces = [
        _build_trace(ts=float(idx), case_id=f"v1_c{idx}", version="v1")
        for idx in range(5)
    ] + [
        _build_trace(ts=float(idx), case_id=f"v2_c{idx}", version="v2")
        for idx in range(5, 10)
    ]

    prepared = _apply_cascade_prepare(
        traces,
        mode="train",
        split_strategy="temporal",
        train_ratio=0.5,
        fraction=1.0,
    )

    all_counts = _trace_version_counts(traces)
    prepared_counts = _trace_version_counts(prepared)

    assert all_counts == {"v1": 5, "v2": 5}
    assert prepared_counts == {"v1": 5}
    assert len(all_counts) > 1 and len(prepared_counts) == 1
    assert _format_trace_version_counts(prepared_counts) == "v1:5"


def test_tensor_scale_diagnostics_flags_unbounded_struct_stats():
    dataset = [
        Data(struct_x=torch.tensor([[0.0, 1.0], [2.0, 6_500_000_000.0]], dtype=torch.float32)),
        Data(struct_x=torch.tensor([[0.0, 3.0], [4.0, 5.0]], dtype=torch.float32)),
    ]

    diagnostics = _tensor_scale_diagnostics(dataset, tensor_name="struct_x", warn_abs_threshold=1_000_000.0)

    assert diagnostics["sampled_graphs"] == 2
    assert diagnostics["values"] == 8
    assert diagnostics["max_abs"] >= 6_400_000_000.0
    assert diagnostics["scale_warning"] is True
