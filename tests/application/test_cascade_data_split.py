from __future__ import annotations

from typing import Iterator

from src.application.use_cases.trainer import ModelTrainer
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


def _build_trace(ts: float, case_id: str) -> RawTrace:
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
    return RawTrace(case_id=case_id, process_version="v1", events=[event], trace_attributes={})


def _trainer(data_config: dict, mode: str = "train") -> ModelTrainer:
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
            "data_config": data_config,
        },
    )


def test_cascade_split_temporal_train_ratio_fraction_then_micro_split():
    traces = [_build_trace(ts=float(ts), case_id=f"c{idx}") for idx, ts in enumerate([5, 1, 4, 0, 3, 2, 9, 7, 8, 6])]
    trainer = _trainer(
        data_config={
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
        data_config={
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
        data_config={
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

