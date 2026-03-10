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
        raise RuntimeError("Not used in window generator tests.")


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


def _trainer(*, window_size: int, sliding: int = 0, stride: int = 0, overlap: int = 0) -> ModelTrainer:
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
            "mode": "eval_drift",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "experiment_config": {
                "drift_window_size": window_size,
                "drift_window_sliding": sliding,
                "drift_window_stride": stride,
                "drift_window_overlap": overlap,
            },
        },
    )


def test_generate_drift_windows_keeps_short_tail_in_legacy_mode():
    traces = [_build_trace(float(i), f"c{i}") for i in range(12)]
    trainer = _trainer(window_size=5, sliding=2)

    windows = trainer._generate_drift_windows(traces)
    starts = [start for start, _ in windows]
    lengths = [len(window) for _, window in windows]

    # Legacy behavior: sliding is ignored, tumbling windows by size with short tail retained.
    assert starts == [0, 5, 10]
    assert lengths == [5, 5, 2]


def test_generate_drift_windows_fallback_to_tumbling_when_sliding_is_zero():
    traces = [_build_trace(float(i), f"c{i}") for i in range(10)]
    trainer = _trainer(window_size=5, sliding=0)

    windows = trainer._generate_drift_windows(traces)
    starts = [start for start, _ in windows]

    assert starts == [0, 5]


def test_generate_drift_windows_uses_legacy_stride_when_configured():
    traces = [_build_trace(float(i), f"c{i}") for i in range(10)]
    trainer = _trainer(window_size=5, stride=2)

    windows = trainer._generate_drift_windows(traces)
    starts = [start for start, _ in windows]

    assert starts == [0, 2, 4, 6, 8]
